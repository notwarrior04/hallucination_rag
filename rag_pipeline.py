"""
Hallucination-Aware RAG Pipeline
=================================
Evidence Highlighting + Contradiction Verification + Verifiable Confidence Scoring

Reference: HaRAG: Hallucination-Aware Retrieval-Augmented Generation
Datasets: SQuAD v2, FEVER, HaluEval
"""

import os
import json
import pandas as pd
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

# Fix for OpenMP multiple initialization error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    pipeline
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from train_models import CombinedHallucinationModel
from temperature_scaling import TemperatureScaler, compute_ece, compute_brier_score

import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Fix 14: Seed control for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────

@dataclass
class RetrievedDocument:
    doc_id: str
    text: str
    score: float
    source: str = ""
    base_doc_id: str = ""
    highlighted_spans: List[Tuple[int, int]] = field(default_factory=list)

@dataclass
class RAGResult:
    query: str
    answer: str
    retrieved_docs: List[RetrievedDocument]
    retrieved_doc_ids: List[str]
    retrieval_scores: List[float]
    evidence_spans: List[Dict]
    contradiction_score: float
    hallucination_probability: float
    hallucination_label: str         # FACTUAL / HALLUCINATED
    verification_score: float
    vcs_score: float
    calibrated_vcs: float
    hallucination_risk: str          # LOW / MEDIUM / HIGH
    verification_label: str          # SUPPORTED / REFUTED / NEI
    confidence_explanation: Dict
    model_versions: Dict             # Metadata for reproducibility
    dataset_name: str = "custom"     # For OOD evaluation tracking
    recall_at_k: Optional[float] = None
    mrr: Optional[float] = None
    hit_rate: Optional[float] = None
    answer_rank_score: Optional[float] = None # Fix 12: For correlation analysis
    component_scores: Dict[str, float] = field(default_factory=dict)
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    composite_halluc_score: float = 0.5  # Multi-signal hallucination score for benchmarks

    def to_dict(self) -> Dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────
# Component 1 — Evidence Highlighter
# ──────────────────────────────────────────────────────────────

class EvidenceHighlighter:
    """
    Identifies and highlights spans in retrieved documents that
    directly support or contradict the generated answer.

    Uses a cross-encoder (fine-tuned on SQuAD v2) for span-level
    relevance scoring.
    """

    def __init__(self, model_name: str = "./checkpoints/evidence_highlighter"):
        # Check if local path exists and contains config.json if it is a directory
        is_valid_local = Path(model_name).exists() and (not Path(model_name).is_dir() or (Path(model_name) / "config.json").exists())
        if not is_valid_local:
            logger.warning(f"Trained highlighter not found or invalid at {model_name}. Using base model.")
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info(f"Loading evidence highlighter: {model_name}")
        self.cross_encoder = CrossEncoder(model_name)
        self.window_size = 3

    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if len(s) > 10]

    def _create_windows(self, sentences: List[str]) -> List[Tuple[str, int, int]]:
        """Sliding window over sentences → (span_text, start_idx, end_idx)."""
        windows = []
        for i in range(len(sentences)):
            for w in range(1, self.window_size + 1):
                end = min(i + w, len(sentences))
                span = " ".join(sentences[i:end])
                windows.append((span, i, end - 1))
        return windows

    def highlight(
        self,
        query: str,
        answer: str,
        doc: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Returns top-k evidence spans with relevance scores.
        Each span dict: {text, sentence_start, sentence_end, score}
        """
        sentences = self._split_sentences(doc)
        if not sentences:
            return []

        windows = self._create_windows(sentences)
        # Adaptive highlighting: use answer for long-form summaries, query for short QA
        is_long_form = len(answer.strip()) > 100
        if is_long_form:
            pairs = [(answer, span) for span, _, _ in windows]
        else:
            # Query-only highlighting to avoid confirmation bias for short QA
            pairs = [(query, span) for span, _, _ in windows]
        scores = self.cross_encoder.predict(pairs, batch_size=32)
        top_indices = np.argsort(scores)[::-1][:top_k]

        evidence = []
        seen = set()
        for idx in top_indices:
            span_text, s_start, s_end = windows[idx]
            key = (s_start, s_end)
            if key not in seen:
                seen.add(key)
                evidence.append({
                    "text": span_text,
                    "sentence_start": s_start,
                    "sentence_end": s_end,
                    "score": float(scores[idx]),
                    "doc_id": "unknown",  # Added by pipeline
                    "source": "unknown"   # Added by pipeline (Fix 8)
                })
        return sorted(evidence, key=lambda x: x["score"], reverse=True)


# ──────────────────────────────────────────────────────────────
# Component 2 — Contradiction Verifier
# ──────────────────────────────────────────────────────────────

class ContradictionVerifier:
    """
    NLI-based contradiction detector fine-tuned on FEVER.
    Labels: ENTAILMENT (0), NEUTRAL (1), CONTRADICTION (2)
    Returns a contradiction probability in [0, 1].
    """

    LABEL_MAP = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}

    def __init__(self, model_name: str = "./checkpoints/contradiction_verifier", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if local path exists and contains config.json if it is a directory
        is_valid_local = Path(model_name).exists() and (not Path(model_name).is_dir() or (Path(model_name) / "config.json").exists())
        if not is_valid_local:
            logger.warning(f"Trained verifier not found or invalid at {model_name}. Using pre-trained NLI base model.")
            # Fall back to a strong pre-trained NLI model if default checkpoint is missing
            if model_name == "./checkpoints/contradiction_verifier":
                model_name = "roberta-large-mnli"
        
        logger.info(f"Loading contradiction verifier model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {model_name}: {e}. Trying use_fast=False.")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            except Exception as e2:
                logger.warning(f"Failed to load with use_fast=False: {e2}. Falling back to roberta-base tokenizer.")
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
                
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.entailment_idx = 0
        self.neutral_idx = 1
        self.contradiction_idx = 2

        if hasattr(self.model.config, "label2id") and self.model.config.label2id:
            label2id = {
                k.lower(): v
                for k, v in self.model.config.label2id.items()
            }

            for label, idx in label2id.items():
                if "entail" in label:
                    self.entailment_idx = idx
                elif "neutral" in label:
                    self.neutral_idx = idx
                elif "contrad" in label:
                    self.contradiction_idx = idx

        self._cache: Dict[str, Dict] = {}

        # Entity resolution setup
        import spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"Could not load spaCy en_core_web_sm model: {e}")
            self.nlp = None

        self.entity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        self.entity_cache = {}
        self.embedding_cache = {}

    def get_entity_embedding(self, text: str):
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        emb = self.entity_model.encode(text, convert_to_tensor=True)
        self.embedding_cache[text] = emb
        return emb

    def _nli_predict(self, premise: str, hypothesis: str) -> Dict:
        key = f"{premise[:80]}|||{hypothesis[:80]}"
        if key in self._cache:
            return self._cache[key]

        with torch.no_grad():
            enc = self.tokenizer(premise, hypothesis, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            out = {
                "entailment": float(probs[self.entailment_idx]),
                "neutral":    float(probs[self.neutral_idx]),
                "contradiction": float(probs[self.contradiction_idx]),
            }

        self._cache[key] = out
        return out

    def check_cheap_match(self, ans_ent: str, ev_ent: str) -> bool:
        ans_lower = ans_ent.lower().strip()
        ev_lower = ev_ent.lower().strip()
        if ans_lower == ev_lower:
            return True
        
        ans_tokens = ans_lower.split()
        ev_tokens = ev_lower.split()
        
        if ans_tokens and ev_tokens:
            if ans_tokens[-1] == ev_tokens[-1] and len(ans_tokens[-1]) >= 3:
                # First name mismatch check
                if len(ans_tokens) > 1 and len(ev_tokens) > 1:
                    if ans_tokens[0] != ev_tokens[0]:
                        return False
                return True
        return False

    def check_answer_grounded_in_text(self, answer: str, text: str) -> bool:
        ans_clean = answer.strip().lower()
        text_clean = text.lower()
        
        # 1. Direct substring match
        if ans_clean in text_clean:
            return True
            
        # 2. Token/word overlap for names/multi-word entities
        ans_words = [w.strip(".,()[]{}") for w in ans_clean.split() if w.strip(".,()[]{}")]
        stopwords = {"and", "or", "the", "of", "in", "on", "at", "a", "an", "for", "with", "by", "to"}
        ans_words = [w for w in ans_words if w not in stopwords]
        
        if not ans_words:
            return False
            
        if len(ans_words) == 1:
            import re
            if re.search(r'\b' + re.escape(ans_words[0]) + r'\b', text_clean):
                return True
                
        if len(ans_words) > 1:
            last_word = ans_words[-1]
            import re
            if re.search(r'\b' + re.escape(last_word) + r'\b', text_clean):
                first_names = ans_words[:-1]
                pattern = re.compile(r'\b(\w+)\s+' + re.escape(last_word) + r'\b')
                matches = pattern.findall(text_clean)
                if matches:
                    for preceding_word in matches:
                        if preceding_word not in stopwords:
                            if preceding_word not in first_names:
                                return False
                return True
        return False

    def check_entity_negated(self, doc_span, ans_ent: str, query_verbs: set) -> bool:
        # Find tokens of ans_ent in doc_span
        ans_tokens = []
        ans_words = set(ans_ent.lower().split())
        for token in doc_span:
            if token.text.lower() in ans_words:
                ans_tokens.append(token)
                
        if not ans_tokens:
            return False
            
        # Find negation heads that match query verbs (or any verb if query_verbs is empty)
        neg_heads = []
        for token in doc_span:
            if token.dep_ == "neg" or token.lower_ in {"not", "never", "n't"}:
                head = token.head
                # Check if the negated head's lemma matches the query verbs
                if not query_verbs or head.lemma_.lower() in query_verbs:
                    neg_heads.append(head)
                else:
                    # Also allow checking ancestors of head if it's an auxiliary
                    curr_head = head
                    while curr_head != curr_head.head:
                        if curr_head.lemma_.lower() in query_verbs:
                            neg_heads.append(head)
                            break
                        curr_head = curr_head.head
                
        if not neg_heads:
            return False
            
        for t in ans_tokens:
            curr = t
            # Traverse up to see if any ancestor is in neg_heads
            while curr != curr.head:
                if curr.head in neg_heads:
                    return True
                curr = curr.head
            # Also check if the token itself is directly the head of a negation (e.g. "not Fleming")
            if t in neg_heads:
                return True
                
        return False

    def check_subject_mismatch(self, doc_span, ans_who_ents: List[str], query_verbs: set, answer_embeddings: dict, entity_embeddings: dict) -> bool:
        if not query_verbs:
            return False
            
        # Find verbs in the span that match query verbs
        matching_verbs = []
        for token in doc_span:
            if token.pos_ in {"VERB", "AUX"} and token.lemma_.lower() in query_verbs:
                matching_verbs.append(token)
                
        if not matching_verbs:
            return False
            
        # For each matching verb, find its subjects/agents
        for verb in matching_verbs:
            subjects = []
            for child in verb.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subjects.append(child)
                elif child.dep_ == "agent": # "by Fleming"
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            subjects.append(grandchild)
                            
            # Filter out pronouns to avoid anaphora false positives
            clean_subjects = [s for s in subjects if s.pos_ != "PRON" and s.lower_ not in {"he", "she", "they", "who", "it", "which", "that"}]
            if not clean_subjects:
                continue
                
            # Convert subjects to string names
            subj_texts = []
            for s in clean_subjects:
                subj_texts.append(s.text.strip())
                if s.doc.noun_chunks:
                    for chunk in s.doc.noun_chunks:
                        if s in chunk:
                            subj_texts.append(chunk.text.strip())
            
            subj_texts = list(set([st for st in subj_texts if st]))
            if not subj_texts:
                continue
                
            has_subject_match = False
            for ans_ent in ans_who_ents:
                for subj_text in subj_texts:
                    if self.check_cheap_match(ans_ent, subj_text):
                        has_subject_match = True
                        break
                    else:
                        if ans_ent not in answer_embeddings:
                            answer_embeddings[ans_ent] = self.get_entity_embedding(ans_ent)
                        emb_ans = answer_embeddings[ans_ent]

                        if subj_text not in entity_embeddings:
                            entity_embeddings[subj_text] = self.get_entity_embedding(subj_text)
                        emb_sub = entity_embeddings[subj_text]

                        sim = torch.nn.functional.cosine_similarity(emb_ans.unsqueeze(0), emb_sub.unsqueeze(0)).item()
                        if sim >= 0.90:
                            has_subject_match = True
                            break
                if has_subject_match:
                    break
                    
            if not has_subject_match:
                logger.info(f"Subject mismatch: verb '{verb.text}' has subject(s) {subj_texts} in evidence, which does not match answer entities {ans_who_ents}.")
                return True
                
        return False

    def filter_redundant_entities(self, entities: List[str]) -> List[str]:
        if not entities:
            return []
        # Sort by length descending
        sorted_ents = sorted(list(set(entities)), key=len, reverse=True)
        filtered = []
        for ent in sorted_ents:
            ent_words = set(ent.lower().split())
            is_subphrase = False
            for existing in filtered:
                existing_words = set(existing.lower().split())
                if ent_words.issubset(existing_words):
                    is_subphrase = True
                    break
            if not is_subphrase:
                filtered.append(ent)
        return filtered

    def get_span_entities(self, text: str) -> List[str]:
        if text in self.entity_cache:
            return self.entity_cache[text]
        
        if self.nlp is None:
            return []
        try:
            doc = self.nlp(text)
            entities = [ent.text.strip() for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "WORK_OF_ART"}]
            entities.extend([chunk.text.strip() for chunk in doc.noun_chunks])
            entities.extend([token.text.strip() for token in doc if token.pos_ == "PROPN"])
            entities = self.filter_redundant_entities(entities)
            self.entity_cache[text] = entities
            return entities
        except Exception as e:
            logger.error(f"Error extracting span entities: {e}")
            return []

    def reconstruct_hypothesis(self, query: str, answer: str) -> str:
        # Verb normalization: map query verbs to forms commonly found in evidence text
        VERB_NORMALIZATIONS = {
            "invented": "developed", "invents": "develops", "invent": "develop",
            "created": "produced", "creates": "produces", "create": "produce",
            "coined": "introduced", "coins": "introduces", "coin": "introduce",
        }
        
        # Strip trailing punctuation from answer to avoid garbled hypotheses like "Delhi. is the capital"
        answer = answer.strip().rstrip(".!?,;:")
        
        query_clean = query.strip("?").strip()
        query_lower = query_clean.lower()
        words = query_lower.split()
        
        if not words:
            return answer
            
        # Preposition extraction, e.g. "In which country was the film..." -> prep = "in"
        prep = ""
        if words[0] in {"in", "on", "at", "during", "by", "from", "with", "about"} and len(words) > 1 and words[1] in {"what", "which"}:
            prep = words[0]
            # Remove the preposition
            query_clean = " ".join(query.split()[1:])
            query_lower = query_clean.lower()
            words = query_lower.split()
            
        first_word = words[0] if words else ""
        
        hyp = answer  # fallback
        
        # 1. Who questions
        if first_word == "who":
            if len(words) > 1 and words[1] in {"is", "was", "are", "were"}:
                hyp = answer.strip() + " " + " ".join(query_clean.split()[1:])
            else:
                hyp = answer.strip() + " " + " ".join(query_clean.split()[1:])
                
        # 2. Where questions
        elif first_word == "where":
            if len(words) > 2 and words[1] == "is":
                hyp = " ".join(query_clean.split()[2:]) + " is in " + answer.strip()
            elif len(words) > 2 and words[1] == "was":
                hyp = " ".join(query_clean.split()[2:]) + " was in " + answer.strip()
            elif len(words) > 2 and words[1] == "are":
                hyp = " ".join(query_clean.split()[2:]) + " are in " + answer.strip()
            elif len(words) > 2 and words[1] == "were":
                hyp = " ".join(query_clean.split()[2:]) + " were in " + answer.strip()
            elif len(words) > 2 and words[1] == "did":
                doc = self.nlp(query_clean)
                root_verb = None
                for token in doc:
                    if token.pos_ == "VERB" and token.i > 1:
                        root_verb = token
                        break
                if root_verb:
                    subj_tokens = [t.text for t in doc[2:root_verb.i]]
                    subj_text = " ".join(subj_tokens).strip()
                    verb_lemma = root_verb.lemma_
                    past_tenses = {"pursue": "pursued", "sink": "sank", "discover": "discovered"}
                    verb_text = past_tenses.get(verb_lemma, root_verb.text)
                    remaining_tokens = [t.text for t in doc[root_verb.i+1:]]
                    remaining_text = " ".join(remaining_tokens).strip()
                    if "toward" in remaining_text:
                        remaining_text = remaining_text.replace("toward", f"toward {answer.strip()}")
                        hyp = f"{subj_text} {verb_text} {remaining_text}"
                    else:
                        hyp = f"{subj_text} {verb_text} {remaining_text} in {answer.strip()}"
                else:
                    hyp = " ".join(query_clean.split()[2:]) + " in " + answer.strip()
                
        # 3. When questions
        elif first_word == "when":
            if len(words) > 2 and words[1] in {"was", "were"}:
                hyp = " ".join(query_clean.split()[2:]) + " " + words[1] + " in " + answer.strip()
            elif len(words) > 2 and words[1] in {"did", "does", "do"}:
                hyp = " ".join(query_clean.split()[2:]) + " in " + answer.strip()
                
        # 4. What/Which questions
        elif first_word in {"what", "which"}:
            if len(words) > 1 and words[1] in {"is", "was", "are", "were"}:
                hyp = answer.strip() + " " + words[1] + " " + " ".join(query_clean.split()[2:])
            else:
                # check for "did/does/do [subject] [verb]" structure
                aux_idx = -1
                aux_word = ""
                for i, w in enumerate(words):
                    if w in {"do", "did", "does"}:
                        aux_idx = i
                        aux_word = w
                        break
                        
                if aux_idx != -1 and aux_idx + 1 < len(words):
                    doc = self.nlp(query_clean)
                    root_verb = None
                    for token in doc:
                        if token.head == token and token.pos_ in {"VERB", "AUX"}:
                            root_verb = token
                            break
                    if not root_verb:
                        for token in doc[aux_idx+1:]:
                            if token.pos_ in {"VERB"}:
                                root_verb = token
                                break
                    if root_verb and root_verb.i > aux_idx:
                        subj_tokens = [t.text for t in doc[aux_idx+1:root_verb.i]]
                        subj_text = " ".join(subj_tokens).strip()
                        
                        verb_lemma = root_verb.lemma_
                        if aux_word == "did":
                            past_tenses = {"develop": "developed", "write": "wrote", "paint": "painted", "produce": "produced", "sink": "sank", "discover": "discovered"}
                            verb_text = past_tenses.get(verb_lemma, root_verb.text)
                        else:
                            verb_text = root_verb.text
                            
                        remaining_tokens = [t.text for t in doc[root_verb.i+1:]]
                        remaining_text = " ".join(remaining_tokens).strip()
                        
                        p_text = f" {prep.lower()} {answer.strip()}" if prep else f" {answer.strip()}"
                        
                        if subj_text and verb_text:
                            if remaining_text:
                                hyp = f"{subj_text} {verb_text}{p_text} {remaining_text}"
                            else:
                                hyp = f"{subj_text} {verb_text}{p_text}"
                
                # If still fallback
                if hyp == answer:
                    doc = self.nlp(query_clean)
                    if len(list(doc.noun_chunks)) > 0:
                        first_chunk = list(doc.noun_chunks)[0]
                        if first_chunk.start == 0:
                            # Truncate first_chunk if it contains a verb
                            end_idx = first_chunk.end
                            for token in first_chunk:
                                if token.pos_ in {"VERB", "AUX"}:
                                    end_idx = token.i
                                    break
                            if end_idx < first_chunk.end:
                                end_char = doc[end_idx].idx
                            else:
                                end_char = first_chunk.end_char
                                
                            p_text = f"{prep.lower()} {answer.strip()}" if prep else answer.strip()
                            hyp = p_text + " " + query_clean[end_char:].strip()
                    
                    # Ultimate fallback
                    if hyp == answer:
                        p_text = f"{prep.lower()} {answer.strip()}" if prep else answer.strip()
                        hyp = p_text + " " + " ".join(query_clean.split()[1:])
                        
        # If a preposition was extracted but not used yet
        if prep and prep.lower() not in hyp.lower():
            if not hyp.endswith("."):
                hyp = hyp.strip() + f" {prep.lower()} {answer.strip()}."
            else:
                hyp = hyp.strip(".").strip() + f" {prep.lower()} {answer.strip()}."
                
        if not hyp.endswith("."):
            hyp += "."
            
        hyp = " ".join(hyp.split())
        hyp = hyp.replace("..", ".").replace("?.", ".")
        
        # Apply verb normalization to the final hypothesis
        hyp_lower = hyp.lower()
        for orig, replacement in VERB_NORMALIZATIONS.items():
            if f" {orig} " in hyp_lower:
                # Case-insensitive replacement preserving surrounding text
                import re
                hyp = re.sub(r'\b' + re.escape(orig) + r'\b', replacement, hyp, flags=re.IGNORECASE)
                break
        
        return hyp

    def verify(
        self,
        answer: str,
        evidence_spans: List[Dict],
        return_entailment: bool = False,
        query: Optional[str] = None
    ) -> Tuple[float, str] | Tuple[float, float, str]:
        """
        Checks each evidence span against the answer.
        Returns (contradiction_score, verification_label) or
        (contradiction_score, entailment_score, verification_label).
        """
        if not evidence_spans:
            if return_entailment:
                return 0.0, 0.0, "NEI"
            return 0.0, "NEI"

        # Determine hypothesis: check if answer is a fragment and query is provided
        hypothesis = answer
        is_entity_answer = False
        if self.nlp is not None:
            try:
                doc_ans = self.nlp(answer)
                is_entity_answer = (
                    len(answer.split()) <= 8 and
                    not any(token.pos_ in {"VERB", "AUX"} for token in doc_ans)
                )
                if is_entity_answer and query:
                    hypothesis = self.reconstruct_hypothesis(query, answer)
                    if hypothesis != answer:
                        logger.info(f"Hypothesis reconstructed: {hypothesis}")
            except Exception as e:
                logger.error(f"Error parsing answer for entity check: {e}")

        # Local caches to optimize embedding retrievals
        answer_embeddings = {}
        entity_embeddings = {}

        # Identify WHO Queries
        is_who_query = False
        ans_who_ents = []
        query_verbs = set()
        if query:
            query_lower = query.strip().lower()
            is_who_query = query_lower.startswith("who")

        if is_who_query and self.nlp is not None:
            try:
                doc_ans = self.nlp(answer)
                ans_who_ents = [ent.text.strip() for ent in doc_ans.ents if ent.label_ in {"PERSON", "ORG"}]
                ans_who_ents.extend([chunk.text.strip() for chunk in doc_ans.noun_chunks])
                ans_who_ents.extend([token.text.strip() for token in doc_ans if token.pos_ == "PROPN"])
                ans_who_ents = self.filter_redundant_entities(ans_who_ents)
                
                # Extract query content verbs to match against negation heads
                if query:
                    doc_query = self.nlp(query)
                    HELPER_LEMMAS = {"be", "do", "have", "can", "will", "shall", "may", "must", "would", "should", "could", "might"}
                    query_verbs = {t.lemma_.lower() for t in doc_query if t.pos_ in {"VERB", "AUX"} and t.lemma_.lower() not in HELPER_LEMMAS}
                    if not query_verbs:
                        query_verbs = {t.lemma_.lower() for t in doc_query if t.pos_ in {"VERB", "AUX"}}
            except Exception as e:
                logger.error(f"Error parsing WHO query/answer entities: {e}")

        contradiction_scores = []
        entailment_scores   = []

        for span in evidence_spans:
            premise = span["text"]
            
            # Resolve aliases in the premise text to improve NLI consistency
            if (is_entity_answer or is_who_query) and self.nlp is not None:
                try:
                    span_entities = self.get_span_entities(premise)
                    # Extract answer entities
                    logging_ans_ents = []
                    if is_who_query and ans_who_ents:
                        logging_ans_ents = ans_who_ents
                    else:
                        doc_ans = self.nlp(answer)
                        logging_ans_ents = [ent.text.strip() for ent in doc_ans.ents if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT"}]
                        logging_ans_ents.extend([chunk.text.strip() for chunk in doc_ans.noun_chunks])
                        logging_ans_ents.extend([token.text.strip() for token in doc_ans if token.pos_ == "PROPN"])
                        logging_ans_ents = self.filter_redundant_entities(logging_ans_ents)
                        if not logging_ans_ents:
                            logging_ans_ents = [answer.strip()]
                            
                    for ans_ent in logging_ans_ents:
                        for ev_ent in span_entities:
                            is_match = False
                            if self.check_cheap_match(ans_ent, ev_ent):
                                is_match = True
                            else:
                                if ans_ent not in answer_embeddings:
                                    answer_embeddings[ans_ent] = self.get_entity_embedding(ans_ent)
                                emb_ans = answer_embeddings[ans_ent]
                                if ev_ent not in entity_embeddings:
                                    entity_embeddings[ev_ent] = self.get_entity_embedding(ev_ent)
                                emb_ev = entity_embeddings[ev_ent]
                                sim = torch.nn.functional.cosine_similarity(emb_ans.unsqueeze(0), emb_ev.unsqueeze(0)).item()
                                if sim >= 0.90:
                                    is_match = True
                                    
                            if is_match and ev_ent.lower() != ans_ent.lower():
                                import re
                                try:
                                    escaped = re.escape(ev_ent)
                                    pattern = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)
                                    new_premise = pattern.sub(ans_ent, premise)
                                    if new_premise == premise:
                                        pattern_simple = re.compile(escaped, re.IGNORECASE)
                                        premise = pattern_simple.sub(ans_ent, premise)
                                    else:
                                        premise = new_premise
                                except Exception:
                                    premise = premise.replace(ev_ent, ans_ent)
                except Exception as e:
                    logger.error(f"Error resolving aliases in premise: {e}")

            scores = self._nli_predict(
                premise=premise,
                hypothesis=hypothesis
            )
            ent_score = scores["entailment"]
            contra_score = scores["contradiction"]

            # Entity Consistency Check and Negation Detection for WHO queries
            if is_who_query and ans_who_ents:
                negation_detected = False
                try:
                    doc_span = self.nlp(premise)
                    for ans_ent in ans_who_ents:
                        if self.check_entity_negated(doc_span, ans_ent, query_verbs):
                            logger.info(f"Negation contradiction detected for entity '{ans_ent}' in span: '{span['text']}'")
                            ent_score = 0.0
                            contra_score = 1.0
                            negation_detected = True
                            break
                except Exception as e:
                    logger.error(f"Error checking negation contradiction: {e}")

                if not negation_detected:
                    span_who_ents = self.get_span_entities(premise)
                    matched_ans_ents = set()
                    
                    for ans_ent in ans_who_ents:
                        found_match = False
                        if self.check_answer_grounded_in_text(ans_ent, span["text"]):
                            found_match = True
                        else:
                            for span_ent in span_who_ents:
                                # 1. Cheap match
                                if self.check_cheap_match(ans_ent, span_ent):
                                    found_match = True
                                    break
                                else:
                                    # 2. Embedding similarity match
                                    if ans_ent not in answer_embeddings:
                                        answer_embeddings[ans_ent] = self.get_entity_embedding(ans_ent)
                                    emb_ans = answer_embeddings[ans_ent]

                                    if span_ent not in entity_embeddings:
                                        entity_embeddings[span_ent] = self.get_entity_embedding(span_ent)
                                    emb_span = entity_embeddings[span_ent]

                                    sim = torch.nn.functional.cosine_similarity(emb_ans.unsqueeze(0), emb_span.unsqueeze(0)).item()
                                    if sim >= 0.70:
                                        found_match = True
                                        break
                        if found_match:
                            matched_ans_ents.add(ans_ent)

                    # Now verify if all answer entities are supported, either directly or as part of a matched larger entity
                    all_answer_entities_supported = True
                    for ans_ent in ans_who_ents:
                        if ans_ent in matched_ans_ents:
                            continue
                        # Check if it is a subphrase of any matched entity
                        is_subphrase_of_matched = False
                        ans_words = set(ans_ent.lower().split())
                        for matched_ent in matched_ans_ents:
                            matched_words = set(matched_ent.lower().split())
                            if ans_words.issubset(matched_words):
                                is_subphrase_of_matched = True
                                break
                        if not is_subphrase_of_matched:
                            all_answer_entities_supported = False
                            break

                    if not all_answer_entities_supported:
                        logger.info(
                            f"Entity consistency mismatch: "
                            f"answer_entities={ans_who_ents}, "
                            f"span_entities={span_who_ents}"
                        )
                        ent_score *= 0.10

            # Calculate entity match score for this span for logging
            span_entity_score = 0.0
            span_entities = self.get_span_entities(premise)
            
            logging_ans_ents = []
            if is_who_query and ans_who_ents:
                logging_ans_ents = ans_who_ents
            elif is_entity_answer:
                try:
                    doc_ans = self.nlp(answer)
                    logging_ans_ents = [ent.text.strip() for ent in doc_ans.ents if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT"}]
                    logging_ans_ents.extend([chunk.text.strip() for chunk in doc_ans.noun_chunks])
                    logging_ans_ents.extend([token.text.strip() for token in doc_ans if token.pos_ == "PROPN"])
                    logging_ans_ents = self.filter_redundant_entities(logging_ans_ents)
                    if not logging_ans_ents:
                        logging_ans_ents = [answer.strip()]
                except Exception:
                    logging_ans_ents = [answer.strip()]
            else:
                logging_ans_ents = [answer.strip()]

            if logging_ans_ents:
                for ans_ent in logging_ans_ents:
                    if self.check_answer_grounded_in_text(ans_ent, span["text"]):
                        span_entity_score = max(span_entity_score, 1.0)
                    for ev_ent in span_entities:
                        if self.check_cheap_match(ans_ent, ev_ent):
                            span_entity_score = max(span_entity_score, 1.0)
                        else:
                            if ans_ent not in answer_embeddings:
                                answer_embeddings[ans_ent] = self.get_entity_embedding(ans_ent)
                            emb_ans = answer_embeddings[ans_ent]
                            if ev_ent not in entity_embeddings:
                                entity_embeddings[ev_ent] = self.get_entity_embedding(ev_ent)
                            emb_ev = entity_embeddings[ev_ent]
                            sim = torch.nn.functional.cosine_similarity(emb_ans.unsqueeze(0), emb_ev.unsqueeze(0)).item()
                            if sim > span_entity_score:
                                span_entity_score = sim

            logger.info(f"Answer: {answer}")
            logger.info(f"Evidence Span: {span['text']}")
            logger.info(f"NLI Score: {scores['entailment']:.4f}")
            logger.info(f"Entity Match Score: {span_entity_score:.4f}")
            logger.info(f"Subject Penalty: 1.0 (Removed)")
            logger.info(f"Final Score: {ent_score:.4f}")

            contradiction_scores.append(contra_score)
            entailment_scores.append(ent_score)

        # Entailment aggregation: blend max with rank-weighted average.
        # This dampens high NLI scores from low-relevance spans (e.g., a Feynman span
        # mentioning "Einstein field equation" getting higher NLI than actual Einstein evidence).
        if entailment_scores:
            max_ent = float(np.max(entailment_scores))
            n = len(entailment_scores)
            # Evidence spans are pre-sorted by highlighter relevance (highest first).
            # Rank-weighted average gives more weight to higher-ranked spans.
            rank_weights = np.array([1.0 / (i + 1) for i in range(n)])
            rank_weights = rank_weights / rank_weights.sum()
            weighted_ent = float(np.dot(entailment_scores, rank_weights))
            avg_entailment = 0.6 * max_ent + 0.4 * weighted_ent
        else:
            avg_entailment = 0.0

        # Compute contradiction score: if there is a strong contradiction in any span, use the maximum contradiction.
        # Otherwise, if it's just background noise, use the contradiction of the best supporting span (the one with max entailment).
        max_contra = float(np.max(contradiction_scores)) if contradiction_scores else 0.0
        if max_contra >= 0.35:
            avg_contradiction = max_contra
        else:
            best_idx = int(np.argmax(entailment_scores)) if entailment_scores else 0
            avg_contradiction = contradiction_scores[best_idx] if contradiction_scores else 0.0

        if avg_entailment > 0.5:
            label = "SUPPORTED"
        elif avg_contradiction > 0.4:
            label = "REFUTED"
        else:
            label = "NEI"

        # Alias Matching & Entity Resolution / Grounding Enforcement
        if is_entity_answer and self.nlp is not None:
            # Extract answer entities
            try:
                doc_ans = self.nlp(answer)
                ans_entities = [ent.text.strip() for ent in doc_ans.ents if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT"}]
                ans_entities.extend([chunk.text.strip() for chunk in doc_ans.noun_chunks])
                ans_entities.extend([token.text.strip() for token in doc_ans if token.pos_ == "PROPN"])
                ans_entities = self.filter_redundant_entities(ans_entities)
                if not ans_entities:
                    ans_entities = [answer.strip()]

                if ans_entities:
                    best_match_score = 0.0
                    best_match_pair = None
                    is_grounded_in_any = False
                    
                    for span in evidence_spans:
                        for ans_ent in ans_entities:
                            if self.check_answer_grounded_in_text(ans_ent, span["text"]):
                                is_grounded_in_any = True
                                break
                        if is_grounded_in_any:
                            break
                            
                    for span in evidence_spans:
                        span_entities = self.get_span_entities(span["text"])
                        for ans_ent in ans_entities:
                            if ans_ent not in answer_embeddings:
                                answer_embeddings[ans_ent] = self.get_entity_embedding(ans_ent)
                            emb_ans = answer_embeddings[ans_ent]
                            
                            for ev_ent in span_entities:
                                # Stage 1: Cheap match
                                if self.check_cheap_match(ans_ent, ev_ent):
                                    best_match_score = max(best_match_score, 1.0)
                                    best_match_pair = (ans_ent, ev_ent)
                                else:
                                    # Stage 2: MiniLM similarity
                                    if ev_ent not in entity_embeddings:
                                        entity_embeddings[ev_ent] = self.get_entity_embedding(ev_ent)
                                    emb_ev = entity_embeddings[ev_ent]
                                    
                                    sim = torch.nn.functional.cosine_similarity(emb_ans.unsqueeze(0), emb_ev.unsqueeze(0)).item()
                                    if sim > best_match_score:
                                        best_match_score = sim
                                        best_match_pair = (ans_ent, ev_ent)
                                        
                    is_grounded_or_matched = is_grounded_in_any or best_match_score >= 0.85
                    
                    if is_grounded_or_matched:
                        if label in {"NEI", "SUPPORTED"} and avg_contradiction < 0.4:
                            if best_match_pair and best_match_score >= 0.85:
                                logger.info(f"Alias match: {best_match_pair[0]} ↔ {best_match_pair[1]} (score={best_match_score:.3f})")
                    else:
                        # Demote if entity was never matched in evidence (Entity absence penalty)
                        is_numeric_match = False
                        # Avoid demoting if the answer (or any part of it) is a number/year that directly matches a substring in the evidence
                        words = answer.strip().split()
                        for w in words:
                            clean_w = w.strip(".,()[]{}")
                            if clean_w.isdigit() and len(clean_w) >= 2:
                                if any(clean_w in span["text"] for span in evidence_spans):
                                    is_numeric_match = True
                                    break
                                    
                        if not is_numeric_match:
                            logger.info(f"Strengthened Entity absence penalty: Scaling down entailment score due to missing entity match (best_match_score={best_match_score:.3f} < 0.85).")
                            avg_entailment = 0.3 * avg_entailment
                            if label == "SUPPORTED":
                                label = "NEI"
            except Exception as e:
                logger.error(f"Error during alias matching / grounding enforcement: {e}")

        if return_entailment:
            return avg_contradiction, avg_entailment, label
        return avg_contradiction, label


# ──────────────────────────────────────────────────────────────
# Component 3 — Hallucination Detector
# ──────────────────────────────────────────────────────────────

class HallucinationDetector:
    """
    Combined classifier that predicts hallucination probability using context, 
    answer, and precomputed component scores.
    """
    def __init__(self, model_path: str = "./checkpoints/confidence_scorer"):
        self.model_path = Path(model_path)
        self._current_path = None
        self._model_cache = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load default path
        self._load_model_for_path(self.model_path)

    def _load_model_for_path(self, path: Path):
        path_str = str(path)
        if path_str in self._model_cache:
            self.model, self.tokenizer, self.ablation_type = self._model_cache[path_str]
            if self.model is not None:
                self.model.to(self.device)
            self._current_path = path
            return

        config_path = path / "config.json"
        if not config_path.exists():
            logger.warning(f"Trained Hallucination Detector config not found at {path}.")
            if path != self.model_path:
                self._load_model_for_path(self.model_path)
            else:
                self.model = None
                self.tokenizer = None
                self.ablation_type = "full"
            return

        with open(config_path, "r") as f:
            config = json.load(f)
        
        ablation_type = config.get("ablation_type", "full")
        base_model = config.get("base_model", "roberta-base")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(path_str)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {path_str}: {e}. Trying use_fast=False.")
            try:
                tokenizer = AutoTokenizer.from_pretrained(path_str, use_fast=False)
            except Exception as e2:
                logger.warning(f"Failed to load with use_fast=False: {e2}. Falling back to roberta-base tokenizer.")
                tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = CombinedHallucinationModel(base_model=base_model, ablation_type=ablation_type)
        
        model_pt = path / "model.pt"
        if model_pt.exists():
            model.load_state_dict(torch.load(str(model_pt), map_location="cpu"))
        
        model.eval()
        model.to(self.device)
        
        self._model_cache[path_str] = (model, tokenizer, ablation_type)
        self.model, self.tokenizer, self.ablation_type = model, tokenizer, ablation_type
        self._current_path = path

    def predict(self, query: str, context: str, answer: str, scores: Dict, mode: str = "full", scorer = None) -> float:
        mode_to_subdir = {
            "baseline_standard": "hallucination_detector_full",
            "baseline_evidence": "hallucination_detector_full",
            "baseline_halluc": "hallucination_detector_full",
            "full": "hallucination_detector_full",
            "evidence_only": "hallucination_detector_evidence_only",
            "verification_only": "hallucination_detector_verification_only",
            "hallucination_only": "hallucination_detector_hallucination_only",
            "hallucination_only_text": "hallucination_detector_hallucination_only_text",
            "evidence_verification": "hallucination_detector_evidence_verification",
            "evidence_hallucination": "hallucination_detector_evidence_hallucination",
            "verification_hallucination": "hallucination_detector_verification_hallucination",
        }

        target_path = self.model_path
        if mode in mode_to_subdir:
            specific_path = self.model_path.parent / mode_to_subdir[mode]
            if (specific_path / "config.json").exists():
                target_path = specific_path

        if self._current_path != target_path:
            logger.info(f"Switching detector to mode checkpoint: {target_path}")
            if self.model is not None:
                try:
                    self.model.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Failed to move model to CPU: {e}")
            self._load_model_for_path(target_path)
            # Sync calibration immediately
            if scorer is not None:
                scorer.load_calibration_for_path(target_path)

        if self.model is None:
            return 0.5
        
        input_text = f"Context: {context[:512]} \n\n Answer: {answer[:256]}"
        enc = self.tokenizer(input_text, truncation=True, max_length=384, padding="max_length", return_tensors="pt").to(self.device)
        
        r = torch.tensor([scores.get("retrieval_quality", 0.0)], dtype=torch.float).to(self.device)
        e = torch.tensor([scores.get("evidence_coverage", 0.0)], dtype=torch.float).to(self.device)
        n = torch.tensor([scores.get("nli_score", 0.0)], dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], 
                             retrieval_scores=r, evidence_scores=e, nli_scores=n)
            logit = out["logits"].item()
            prob = 1.0 / (1.0 + np.exp(-logit))
        return float(prob)


# ──────────────────────────────────────────────────────────────
# Component 4 — Verifiable Confidence Scorer
# ──────────────────────────────────────────────────────────────

# Fix 15: Meta-model for learned VCS
class VCSMetaModel(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class VerifiableConfidenceScorer:
    """
    Combines four signals into a calibrated VCS score.
    Uses temperature scaling for calibration.
    """

    def __init__(self, config_path: str, temperature_path: str, meta_model_path: Optional[str] = None):
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            self.thresholds = config.get("thresholds", {"low_risk": 0.75, "medium_risk": 0.50})
            self.weights = config.get("weights", {"retrieval": 0.25, "evidence": 0.25, "verification": 0.25, "hallucination": 0.25})
        else:
            self.thresholds = {"low_risk": 0.75, "medium_risk": 0.50}
            self.weights = {"retrieval": 0.25, "evidence": 0.25, "verification": 0.25, "hallucination": 0.25}

        self.scaler = TemperatureScaler()
        # Fix 13: Temperature loading robustness
        if Path(temperature_path).exists():
            try:
                with open(temperature_path, "r") as f:
                    t_config = json.load(f)
                self.scaler.temperature = float(t_config.get("temperature", 1.0))
                logger.info(f"Loaded calibration temperature: {self.scaler.temperature}")
            except Exception as e:
                logger.warning(f"Failed to load temperature from {temperature_path}: {e}. Defaulting to 1.0")
                self.scaler.temperature = 1.0
        else:
            self.scaler.temperature = 1.0

        # Fix 1: Load Meta-Model for VCS
        self.meta_model = None
        if meta_model_path and Path(meta_model_path).exists():
            try:
                self.meta_model = VCSMetaModel()
                self.meta_model.load_state_dict(torch.load(meta_model_path, map_location="cpu"))
                self.meta_model.eval()
                logger.info(f"Loaded trained VCS meta-model from {meta_model_path}")
            except Exception as e:
                logger.error(f"Failed to load VCS meta-model: {e}")

        # Fix 5: Load hallucination threshold
        self.halluc_threshold = 0.5
        threshold_path = Path(temperature_path).parent / "hallucination_threshold.json"
        if threshold_path.exists():
            try:
                with open(threshold_path, "r") as f:
                    h_config = json.load(f)
                self.halluc_threshold = float(h_config.get("threshold", 0.5))
                logger.info(f"Loaded hallucination threshold: {self.halluc_threshold}")
            except: pass

    def load_calibration_for_path(self, path: Path):
        temperature_path = path / "temperature.json"
        meta_model_path = path / "vcs_meta_model.pt"
        threshold_path = path / "hallucination_threshold.json"

        # Load temperature
        if temperature_path.exists():
            try:
                with open(temperature_path, "r") as f:
                    t_config = json.load(f)
                self.scaler.temperature = float(t_config.get("temperature", 1.0))
                logger.info(f"Loaded calibration temperature from {path}: {self.scaler.temperature}")
            except Exception as e:
                logger.warning(f"Failed to load temperature from {temperature_path}: {e}. Defaulting to 1.0")
                self.scaler.temperature = 1.0
        else:
            self.scaler.temperature = 1.0

        # Load meta model
        self.meta_model = None
        if meta_model_path.exists():
            try:
                self.meta_model = VCSMetaModel()
                self.meta_model.load_state_dict(torch.load(str(meta_model_path), map_location="cpu"))
                self.meta_model.eval()
                logger.info(f"Loaded trained VCS meta-model from {meta_model_path}")
            except Exception as e:
                logger.error(f"Failed to load VCS meta-model: {e}")

        # Load threshold
        self.halluc_threshold = 0.5
        if threshold_path.exists():
            try:
                with open(threshold_path, "r") as f:
                    h_config = json.load(f)
                self.halluc_threshold = float(h_config.get("threshold", 0.5))
                logger.info(f"Loaded hallucination threshold from {path}: {self.halluc_threshold}")
            except: pass

    def score(
        self,
        retrieval_quality: float,
        evidence_support: float,
        verification_score: float,
        hallucination_probability: float
    ) -> Tuple[float, float, Dict]:
        
        # Fix 15: Option to use learned meta-model instead of fixed weights
        if hasattr(self, "meta_model") and self.meta_model is not None:
            features = torch.tensor([[retrieval_quality, evidence_support, verification_score, 1.0 - hallucination_probability]], dtype=torch.float)
            with torch.no_grad():
                vcs = float(self.meta_model(features).squeeze())
        else:
            # Fallback to hand-crafted weights
            vcs = (
                self.weights["retrieval"] * retrieval_quality +
                self.weights["evidence"] * evidence_support +
                self.weights["verification"] * verification_score +
                self.weights["hallucination"] * (1.0 - hallucination_probability)
            )
        
        # Fix 1: Dangerous VCS calibration (numerical stability)
        eps = 1e-8
        vcs_clamped = np.clip(vcs, eps, 1.0 - eps)
        logit = np.log(vcs_clamped / (1.0 - vcs_clamped))
        calibrated_logit = logit / self.scaler.temperature
        calibrated_vcs = 1.0 / (1.0 + np.exp(-calibrated_logit))
        
        # Explanation
        if vcs >= self.thresholds["low_risk"]:
            reason = "High confidence: Strong retrieval and evidence support with low contradiction risk."
        elif vcs >= self.thresholds["medium_risk"]:
            reason = "Moderate confidence: Some evidence coverage but potential gaps or minor contradictions."
        else:
            reason = "Low confidence: Poor evidence support or high hallucination probability detected."

        explanation = {
            "retrieval_quality": round(retrieval_quality, 4),
            "evidence_support": round(evidence_support, 4),
            "verification_score": round(verification_score, 4),
            "hallucination_probability": round(hallucination_probability, 4),
            "vcs": round(vcs, 4),
            "calibrated_vcs": round(calibrated_vcs, 4),
            "reason": reason,
            "hand_crafted_weights": self.weights # Meta-data
        }
        
        return float(vcs), float(calibrated_vcs), explanation

    def risk_level(self, vcs: float) -> str:
        if vcs >= self.thresholds["low_risk"]:
            return "LOW"
        elif vcs >= self.thresholds["medium_risk"]:
            return "MEDIUM"
        else:
            return "HIGH"


# ──────────────────────────────────────────────────────────────
# Retriever (Dense + BM25 hybrid)
# ──────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if start + step >= len(words):
            break
        start += step
    return chunks


def clean_for_bm25(text: str) -> List[str]:
    import re
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


class HybridRetriever:
    """
    Dense bi-encoder retrieval fused with BM25 sparse retrieval.
    Corpus can be loaded from SQuAD v2 / FEVER contexts.
    """

    def __init__(self, bi_encoder_name: str = "BAAI/bge-small-en-v1.5", chunking_config: Optional[Dict] = None, fusion_method: str = "weighted"):
        self.bi_encoder_name = bi_encoder_name
        self.bi_encoder = SentenceTransformer(bi_encoder_name)
        
        # Load Cross-Encoder reranker
        cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info(f"Loading retriever cross-encoder: {cross_encoder_name}")
        self.cross_encoder = CrossEncoder(cross_encoder_name)

        self.corpus: List[Dict] = []
        self.corpus_embeddings = None
        self._bm25 = None
        
        self.chunking_config = chunking_config
        self.fusion_method = fusion_method

    def index(self, documents: List[Dict]):
        """
        documents: list of {doc_id, text, source, ...}
        """
        logger.info(f"Indexing {len(documents)} documents...")
        
        if self.chunking_config is not None:
            chunk_size = self.chunking_config.get("chunk_size", 200)
            overlap = self.chunking_config.get("overlap", 50)
            
            chunked_docs = []
            chunk_lengths = []
            for doc in documents:
                chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
                for i, chunk in enumerate(chunks):
                    meta_parts = []
                    title = doc.get("title", "")
                    if title:
                        meta_parts.append(title.replace("_", " "))
                    claim = doc.get("claim", "")
                    if claim:
                        meta_parts.append(claim)
                        
                    if meta_parts:
                        meta_prefix = " | ".join(meta_parts)
                        chunk_text_with_meta = f"{meta_prefix} - {chunk}"
                    else:
                        chunk_text_with_meta = chunk
                        
                    chunked_docs.append({
                        "doc_id": f"{doc['doc_id']}__chunk_{i}",
                        "base_doc_id": doc["doc_id"],
                        "chunk_id": f"chunk_{i}",
                        "text": chunk_text_with_meta,
                        "source": doc.get("source", ""),
                        "title": doc.get("title", ""),
                        "claim": doc.get("claim", "")
                    })
                    chunk_lengths.append(len(chunk_text_with_meta.split()))
            
            self.corpus = chunked_docs
            num_docs = len(documents)
            num_chunks = len(chunked_docs)
            avg_words = np.mean(chunk_lengths) if chunk_lengths else 0.0
            min_words = np.min(chunk_lengths) if chunk_lengths else 0
            max_words = np.max(chunk_lengths) if chunk_lengths else 0
            logger.info(f"Chunked {num_docs} docs into {num_chunks} chunks (avg={avg_words:.1f}, min={min_words}, max={max_words} words)")
        else:
            # Baseline: No chunking
            self.corpus = []
            for doc in documents:
                meta_parts = []
                title = doc.get("title", "")
                if title:
                    meta_parts.append(title.replace("_", " "))
                claim = doc.get("claim", "")
                if claim:
                    meta_parts.append(claim)
                    
                if meta_parts:
                    meta_prefix = " | ".join(meta_parts)
                    doc_text = f"{meta_prefix} - {doc['text']}"
                else:
                    doc_text = doc["text"]
                    
                self.corpus.append({
                    "doc_id": doc["doc_id"],
                    "base_doc_id": doc["doc_id"],
                    "chunk_id": 0,
                    "text": doc_text,
                    "source": doc.get("source", ""),
                    "title": doc.get("title", ""),
                    "claim": doc.get("claim", "")
                })
            logger.info(f"Indexed {len(documents)} documents without chunking.")

        texts = [d["text"] for d in self.corpus]

        # Dense index
        self.corpus_embeddings = self.bi_encoder.encode(
            texts, batch_size=16, convert_to_tensor=True, show_progress_bar=True
        )

        # BM25 sparse index
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [clean_for_bm25(t) for t in texts]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("BM25 index built.")
        except ImportError:
            logger.warning("rank_bm25 not installed; using dense-only retrieval.")

    def _get_core_query_entities(self, query: str) -> List[str]:
        import spacy
        try:
            if not hasattr(self, "_nlp") or self._nlp is None:
                self._nlp = spacy.load("en_core_web_sm")
            doc = self._nlp(query)
            entities = []
            for ent in doc.ents:
                if ent.label_ not in {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}:
                    entities.append(ent.text.lower().strip())
            
            for token in doc:
                if token.pos_ in {"PROPN", "NOUN"} and token.lemma_.lower() not in {
                    "who", "what", "where", "when", "why", "how", "is", "was", "are", "were", 
                    "did", "does", "do", "the", "a", "an", "of", "in", "to", "for", "with", 
                    "on", "at", "by", "from", "and", "or", "about", "during", "first",
                    "scientist", "discovery", "invention", "inventor", "author", "writer", 
                    "painter", "painting", "founder", "foundation", "development", "developer"
                }:
                    entities.append(token.text.lower().strip())
                    entities.append(token.lemma_.lower().strip())
            
            clean_entities = list(set([e for e in entities if len(e) > 1]))
            return clean_entities
        except Exception as e:
            logger.warning(f"Error extracting core query entities: {e}")
            words = query.lower().strip("?. ").split()
            stopwords = {
                "who", "what", "where", "when", "why", "how", "is", "was", "are", "were", 
                "did", "does", "do", "the", "a", "an", "of", "in", "to", "for", "with", 
                "on", "at", "by", "from", "and", "or", "about", "during", "first",
                "scientist", "discovery", "invention", "inventor", "author", "writer", 
                "painter", "painting", "founder", "foundation", "development", "developer"
            }
            return [w for w in words if w not in stopwords and len(w) > 2]

    def retrieve(self, query: str, top_k: int = 20) -> Tuple[List[RetrievedDocument], np.ndarray]:
        if self.corpus_embeddings is None:
            raise RuntimeError("Call .index() before .retrieve()")

        # Generate deterministic expansions (disabled temporarily)
        expansions = []
        query_lower = query.lower().strip("?. ")
        
        if False:
            # 1. Who discovered X -> discovery of X, scientist who discovered X, discoverer of X
            if "discovered " in query_lower:
                parts = query_lower.split("discovered ")
                if len(parts) > 1 and parts[1].strip():
                    X = parts[1].strip()
                    expansions.append(f"discovery of {X}")
                    expansions.append(f"scientist who discovered {X}")
                    expansions.append(f"discoverer of {X}")
                    
            # 2. Who invented X -> invention of X, inventor of X
            elif "invented " in query_lower:
                parts = query_lower.split("invented ")
                if len(parts) > 1 and parts[1].strip():
                    X = parts[1].strip()
                    expansions.append(f"invention of {X}")
                    expansions.append(f"inventor of {X}")
                    
            # 3. Who developed X -> development of X, developer of X
            elif "developed " in query_lower:
                parts = query_lower.split("developed ")
                if len(parts) > 1 and parts[1].strip():
                    X = parts[1].strip()
                    expansions.append(f"development of {X}")
                    expansions.append(f"developer of {X}")
                    
            # 4. Who wrote X -> author of X, writer of X, written by X
            elif "wrote " in query_lower:
                parts = query_lower.split("wrote ")
                if len(parts) > 1 and parts[1].strip():
                    X = parts[1].strip()
                    expansions.append(f"author of {X}")
                    expansions.append(f"writer of {X}")
                    expansions.append(f"written by {X}")
                    
            # 5. Who painted X -> painter of X, painting of X
            elif "painted " in query_lower:
                parts = query_lower.split("painted ")
                if len(parts) > 1 and parts[1].strip():
                    X = parts[1].strip()
                    expansions.append(f"painter of {X}")
                    expansions.append(f"painting of {X}")
                    
            # 6. Who founded X -> founder of X, foundation of X
            elif "founded " in query_lower:
                parts = query_lower.split("founded ")
                if len(parts) > 1 and parts[1].strip():
                    X = parts[1].strip()
                    expansions.append(f"founder of {X}")
                    expansions.append(f"foundation of {X}")
                    
            # 7. Who is the first Y / was first Y
            if "first " in query_lower:
                parts = query_lower.split("first ")
                if len(parts) > 1 and parts[1].strip():
                    Y = parts[1].strip()
                    expansions.append(f"first {Y}")
                    if "was the first " in query_lower:
                        expansions.append(Y)

        if expansions:
            logger.info(f"Generated query expansions for '{query}': {expansions}")

        # Prepend instruction prefix for BGE models
        query_for_dense = query
        if hasattr(self, "bi_encoder_name") and "bge" in self.bi_encoder_name.lower():
            query_for_dense = f"Represent this sentence for searching relevant passages: {query}"

        q_emb = self.bi_encoder.encode(query_for_dense, convert_to_tensor=True)
        
        # In case we have expansions, average their dense representations
        if expansions:
            q_embs = [q_emb]
            for exp in expansions:
                exp_dense = exp
                if hasattr(self, "bi_encoder_name") and "bge" in self.bi_encoder_name.lower():
                    exp_dense = f"Represent this sentence for searching relevant passages: {exp}"
                q_embs.append(self.bi_encoder.encode(exp_dense, convert_to_tensor=True))
            q_emb = torch.stack(q_embs).mean(dim=0)

        dense_scores = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), self.corpus_embeddings
        ).cpu().numpy()

        # Fuse scores
        if self._bm25 is not None:
            # BM25 gets original query and expansions combined
            bm25_query_text = " ".join([query] + expansions)
            bm25_scores = np.array(self._bm25.get_scores(clean_for_bm25(bm25_query_text)))
            
            if self.fusion_method == "rrf":
                RRF_K = 60
                dense_order = np.argsort(dense_scores)[::-1]
                dense_ranks = np.empty_like(dense_order)
                dense_ranks[dense_order] = np.arange(1, len(dense_order) + 1)
                
                bm25_order = np.argsort(bm25_scores)[::-1]
                bm25_ranks = np.empty_like(bm25_order)
                bm25_ranks[bm25_order] = np.arange(1, len(bm25_order) + 1)
                
                fused = 1.0 / (RRF_K + dense_ranks) + 1.0 / (RRF_K + bm25_ranks)
                
                fused_k = max(150, top_k)
                rerank_k = max(75, top_k)
            else:
                # Weighted score fusion
                def norm(x):
                    rng = x.max() - x.min()
                    return (x - x.min()) / (rng + 1e-9)
                fused = 0.6 * norm(dense_scores) + 0.4 * norm(bm25_scores)
                
                fused_k = max(100, top_k)
                rerank_k = max(50, top_k)
        else:
            fused = dense_scores
            if self.fusion_method == "rrf":
                fused_k = max(150, top_k)
                rerank_k = max(75, top_k)
            else:
                fused_k = max(100, top_k)
                rerank_k = max(50, top_k)

        fused_sorted_indices = np.argsort(fused)[::-1]
        fused_candidates = fused_sorted_indices[:fused_k]
        rerank_candidates = fused_candidates[:rerank_k]
        
        # Then rerank top candidates
        pairs = [
            (query, self.corpus[idx]["text"])
            for idx in rerank_candidates
        ]

        rerank_scores = self.cross_encoder.predict(pairs)

        # Apply soft grounding penalty
        core_entities = self._get_core_query_entities(query)
        if core_entities:
            adjusted_scores = []
            for score, idx in zip(rerank_scores, rerank_candidates):
                doc_text = self.corpus[idx]["text"]
                doc_text_lower = doc_text.lower()
                has_match = False
                for ent in core_entities:
                    if len(ent.split()) > 1:
                        if ent in doc_text_lower:
                            has_match = True
                            break
                    else:
                        import re
                        if re.search(r'\b' + re.escape(ent) + r'\b', doc_text_lower):
                            has_match = True
                            break
                if not has_match:
                    new_score = score - 3.0
                    adjusted_scores.append(new_score)
                else:
                    adjusted_scores.append(score)
            rerank_scores = adjusted_scores

        reranked = sorted(
            zip(rerank_candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Reranked top indices
        top_indices_and_scores = reranked[:top_k]
        top_indices = [x[0] for x in top_indices_and_scores]
        
        results = []
        for idx, score in top_indices_and_scores:
            doc = self.corpus[idx]
            results.append(RetrievedDocument(
                doc_id=doc["doc_id"],
                base_doc_id=doc.get("base_doc_id", doc["doc_id"]),
                text=doc["text"],
                score=float(score),
                source=doc.get("source", "")
            ))
        return results, dense_scores[top_indices]


# ──────────────────────────────────────────────────────────────
# Generator (wraps any HF causal LM or seq2seq)
# ──────────────────────────────────────────────────────────────

class Generator:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        logger.info(f"Loading generator: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def generate(self, query: str, context: str) -> str:
        # Format passages as numbered list for better comprehension
        passages = [p.strip() for p in context.split("\n\n") if p.strip()]
        formatted_passages = "\n".join(
            f"[{i+1}] {p}" for i, p in enumerate(passages[:10])
        )

        prompt = (
            "You are a factual question-answering system. "
            "Answer the question using ONLY the information in the provided passages. "
            "If multiple passages discuss the topic, prefer information that is consistent across passages. "
            "Ignore any single passage that makes a claim contradicted by other passages. "
            "If the passages do not contain enough information to answer, say 'I don't know'.\n\n"
            f"Passages:\n{formatted_passages}\n\n"
            f"Question: {query}\n\n"
            "Answer (be concise, give only the answer):"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()


# ──────────────────────────────────────────────────────────────
# Master Pipeline
# ──────────────────────────────────────────────────────────────

class HallucinationAwareRAG:
    """
    Full pipeline:
      Retrieval → Generation → Evidence Highlighting
      → Contradiction Verification → Confidence Scoring
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[Generator] = None,
        highlighter_path: Optional[str] = None,
        verifier_path: Optional[str] = None,
        hallucination_path: str = "./checkpoints/confidence_scorer",
        config_path: str = "configs/vcs_config.json"
    ):
        highlighter_path = highlighter_path or "./checkpoints/evidence_highlighter"
        verifier_path    = verifier_path    or "./checkpoints/contradiction_verifier"
        hallucination_path = hallucination_path or "./checkpoints/confidence_scorer"
        
        self.retriever   = retriever   or HybridRetriever(
            chunking_config={"chunk_size": 128, "overlap": 32},
            fusion_method="rrf"
        )
        self.generator   = generator   or Generator(model_name)
        self.highlighter = EvidenceHighlighter(highlighter_path)
        self.verifier    = ContradictionVerifier(verifier_path)
        self.detector    = HallucinationDetector(hallucination_path)
        
        temperature_path = Path(hallucination_path) / "temperature.json"
        meta_model_path  = Path(hallucination_path) / "vcs_meta_model.pt"
        
        self.scorer      = VerifiableConfidenceScorer(config_path, str(temperature_path), str(meta_model_path) if meta_model_path.exists() else None)
        
        # Save paths for metadata (Fix 9)
        self.model_paths = {
            "evidence": str(highlighter_path),
            "verifier": str(verifier_path),
            "hallucination": str(hallucination_path)
        }

    def run(self, query: str, top_k: int = 5, mode: str = "full", dataset_name: str = "custom", gold_doc_id: Optional[str] = None, pre_generated_answer: Optional[str] = None) -> RAGResult:
        # Step 1: Retrieve a larger pool to allow deduplication to top_k unique documents
        retrieval_limit = max(50, top_k * 4)
        chunks, top_dense_scores = self.retriever.retrieve(query, top_k=retrieval_limit)
        
        # Deduplicate chunks to base documents & unique text content
        seen_base_ids = set()
        seen_texts = set()
        dedup_docs = []
        for d in chunks:
            text_norm = " ".join(d.text.lower().split())
            # Strip any metadata prefix (e.g. "Title - " or "Title | Claim - ") for accurate comparison
            if " - " in text_norm:
                text_norm = text_norm.split(" - ", 1)[-1]
                
            if d.base_doc_id not in seen_base_ids and text_norm not in seen_texts:
                seen_base_ids.add(d.base_doc_id)
                seen_texts.add(text_norm)
                dedup_docs.append(d)
        
        docs = dedup_docs[:top_k]
        retrieved_ids = [d.doc_id for d in docs]
        
        # Compute top scores based on deduplicated documents
        top_doc_score = float(docs[0].score) if docs else 0.0
        mean_top5_score = float(np.mean([d.score for d in docs[:5]])) if docs else 0.0
        
        best_rank = -1
        if gold_doc_id is not None:
            for i, d in enumerate(dedup_docs):
                if d.base_doc_id == gold_doc_id or d.doc_id == gold_doc_id:
                    best_rank = i + 1
                    break
        
        # Issue 2: Fixed retrieval quality to match train_models.py (cosine similarity of query vs context)
        # Note: Truncated to 512 chars as in training for perfect feature alignment
        # Clean chunks of any metadata prefix like "ragtruth - " or "Title | Claim - " to match training context format
        clean_doc_texts = []
        for d in docs:
            t = d.text
            if " - " in t:
                # Strip metadata prefix before the first ' - '
                t = t.split(" - ", 1)[-1]
            clean_doc_texts.append(t)
        context = "\n\n".join(clean_doc_texts)
        with torch.no_grad():
            q_emb = self.retriever.bi_encoder.encode(query, convert_to_tensor=True)
            c_emb = self.retriever.bi_encoder.encode(context[:512], convert_to_tensor=True)
            retrieval_quality = float(torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), c_emb.unsqueeze(0)).cpu().numpy()[0])
            retrieval_quality = float(np.clip(retrieval_quality, 0.0, 1.0))

        # Step 2: Generate
        if pre_generated_answer is not None:
            answer = pre_generated_answer
        else:
            answer = self.generator.generate(query, context)
        
        # Post-process: expand abbreviated answers to full entities found in top documents.
        # FLAN-T5 often truncates "New Delhi" → "Delhi." or "George Washington" → "Washington".
        # If the stripped answer matches the SUFFIX of a named entity in the context, expand it.
        if answer and len(answer.strip()) <= 50:
            import re as _re
            ans_stripped = answer.strip().rstrip(".!?,;:")
            ans_lower = ans_stripped.lower()
            # Search all top docs for a multi-word entity whose last word matches the answer
            candidate_expansions = []
            for doc in docs[:3]:
                doc_text = doc.text
                if " - " in doc_text:
                    # Extract metadata prefix (e.g. "New Delhi")
                    prefix = doc_text.split(" - ", 1)[0].strip()
                    if prefix.lower().endswith(ans_lower) and len(prefix) > len(ans_stripped):
                        candidate_expansions.append(prefix)
                # Also search for capitalized multi-word phrases ending with the answer
                # e.g. "New Delhi" from text containing "New Delhi"
                pattern = _re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+' + _re.escape(ans_stripped) + r')\b')
                matches = pattern.findall(doc_text)
                candidate_expansions.extend(matches)
            # Pick shortest valid expansion (most precise)
            if candidate_expansions:
                valid = [c for c in candidate_expansions if len(c) > len(ans_stripped) and len(c) <= len(ans_stripped) + 20]
                if valid:
                    expanded = min(valid, key=len)
                    logger.info(f"Expanded answer '{ans_stripped}' → '{expanded}'")
                    answer = expanded


        # Compute answer-context semantic similarity (important for long-form responses)
        # Factual summaries should be semantically closer to the source than hallucinated ones
        with torch.no_grad():
            a_emb = self.retriever.bi_encoder.encode(answer[:512], convert_to_tensor=True)
            chunk_sims = []
            for d in docs:
                c_emb_chunk = self.retriever.bi_encoder.encode(d.text[:1024], convert_to_tensor=True)
                sim_val = float(torch.nn.functional.cosine_similarity(a_emb.unsqueeze(0), c_emb_chunk.unsqueeze(0)).cpu().numpy()[0])
                chunk_sims.append(sim_val)
            answer_context_sim = max(chunk_sims) if chunk_sims else 0.0
            answer_context_sim = float(np.clip(answer_context_sim, 0.0, 1.0))

        # Step 3: Evidence Highlighting
        all_evidence = []
        if mode in ("full", "evidence_only", "verification_only", "hallucination_only", "hallucination_only_text", "evidence_hallucination", "verification_hallucination", "evidence_verification", "baseline_evidence", "baseline_halluc"):
            for doc in docs:
                spans = self.highlighter.highlight(query, answer, doc.text, top_k=2)
                for span in spans:
                    span["doc_id"] = doc.doc_id
                    span["source"] = doc.source  # Fix 8: Attribution
                    # Blend individual span score with doc retrieval score (User Fix 2)
                    span["score"] = 0.7 * span["score"] + 0.3 * doc.score
                all_evidence.extend(spans)
        
        # Deduplicate and keep top evidence
        all_evidence = sorted(all_evidence, key=lambda x: x["score"], reverse=True)[:5]
        if all_evidence:
            max_score = all_evidence[0]["score"]
            all_evidence = [e for e in all_evidence if e["score"] >= max(1.5, 0.4 * max_score)]
        
        # Fix: Boost evidence spans that directly contain the answer text — helps NLI verification
        # for short extractive answers (e.g. "New Delhi") which may not entail longer spans.
        # Also handle cases where answer is a sub-word of entity in evidence ("Delhi" in "New Delhi").
        if all_evidence and answer and len(answer.strip()) <= 100:
            answer_clean_lower = " ".join(answer.lower().strip().rstrip(".").split())
            boosted = []
            normal = []
            for e in all_evidence:
                ev_lower = e["text"].lower()
                # Match if answer (or any word of answer) is present in evidence
                if answer_clean_lower in ev_lower or any(
                    word in ev_lower for word in answer_clean_lower.split() if len(word) > 3
                ):
                    boosted.append(e)
                else:
                    normal.append(e)
            # Put direct-grounding spans first so NLI has best chance of entailment
            all_evidence = boosted + normal
        # Fix 3: Evidence score normalization (matching train_models.py:582)
        raw_evidence_mean = float(np.mean([e["score"] for e in all_evidence])) if all_evidence else 0.0
        evidence_support = float(1.0 / (1.0 + np.exp(-raw_evidence_mean))) if all_evidence else 0.0
        
        # Blend evidence support with retrieval quality (User Fix 2)
        evidence_support = 0.7 * evidence_support + 0.3 * retrieval_quality

        # Compute answer-evidence semantic similarity
        with torch.no_grad():
            evidence_text = " ".join([e["text"] for e in all_evidence])
            if evidence_text:
                e_emb = self.retriever.bi_encoder.encode(evidence_text[:1024], convert_to_tensor=True)
                answer_evidence_sim = float(torch.nn.functional.cosine_similarity(a_emb.unsqueeze(0), e_emb.unsqueeze(0)).cpu().numpy()[0])
                answer_evidence_sim = float(np.clip(answer_evidence_sim, 0.0, 1.0))
            else:
                answer_evidence_sim = 0.0

        # Step 4: Contradiction Verification
        # Filter out FEVER REFUTES-labeled spans from NLI evidence to avoid false REFUTED verdicts.
        # FEVER corpus includes claims like "Telangana is an Indian capital" (a REFUTES claim) that
        # can incorrectly trigger contradiction when checked against correct factual answers.
        if mode in ("full", "verification_only", "hallucination_only", "hallucination_only_text", "verification_hallucination", "evidence_verification", "baseline_halluc"):
            # Separate SQuAD-sourced from FEVER-sourced evidence for NLI
            squad_evidence = [e for e in all_evidence if e.get("source", "") == "squad_v2"]
            fever_evidence = [e for e in all_evidence if e.get("source", "") == "fever"]
            # Prefer squad evidence for NLI if available, otherwise use all evidence
            nli_evidence = squad_evidence if squad_evidence else all_evidence
            contradiction_score, entailment_score, verification_label = self.verifier.verify(answer, nli_evidence, return_entailment=True, query=query)
            # If squad gave NEI but FEVER has supportive content, re-check with all evidence
            if verification_label == "NEI" and fever_evidence:
                c2, e2, l2 = self.verifier.verify(answer, all_evidence, return_entailment=True, query=query)
                if l2 == "SUPPORTED":
                    contradiction_score, entailment_score, verification_label = c2, e2, l2
        else:
            contradiction_score, entailment_score, verification_label = 0.0, 0.0, "NEI"
        
        # Use absolute NLI entailment difference to handle NEI vs SUPPORT (User Fix 1 revised)
        verification_score = max(0.0, entailment_score - contradiction_score)

        # Step 4.5: Answer Abstention
        # Force "I don't know" only when BOTH evidence_support AND verification_score are low,
        # AND the answer is not directly found in any evidence span (direct grounding).
        # Using OR caused false abstentions for factual answers grounded in retrieved passages.
        if pre_generated_answer is None and mode in ("full", "verification_only", "evidence_verification"):
            answer_lower = answer.lower()
            if not ("don't know" in answer_lower or "dont know" in answer_lower or "cannot" in answer_lower):
                # Check direct grounding: is the answer text present in any evidence span?
                # Use stripped/normalized form and partial word match for robustness.
                answer_clean_for_ground = " ".join(answer_lower.rstrip(".").split())
                is_directly_grounded = any(
                    answer_clean_for_ground in e["text"].lower() or
                    any(word in e["text"].lower() for word in answer_clean_for_ground.split() if len(word) > 3)
                    for e in all_evidence
                ) if all_evidence else False
                
                # Abstain only when: BOTH scores are weak AND answer is not directly grounded
                should_abstain = (
                    not is_directly_grounded
                    and verification_score < 0.10
                    and evidence_support < 0.20
                )
                if should_abstain:
                    logger.warning(f"Abstaining from answer: verification_score ({verification_score:.4f}) and evidence_support ({evidence_support:.4f}) are both too low and answer not grounded.")
                    answer = "I don't know"
                    # Reset verification metrics for the abstained answer
                    contradiction_score, entailment_score, verification_label = 0.0, 0.0, "NEI"
                    verification_score = 0.0

        # Step 5: Hallucination Detection
        # Issue 4: For a true hallucination ablation, dependent features should be 0 if the model supports it.
        # However, for consistency with our fixed-architecture baseline, we run them but allow VCS to use only the detector score.
        detector_input_scores = {
            "retrieval_quality": retrieval_quality,
            "evidence_coverage": evidence_support,
            "nli_score": contradiction_score
        }
        
        # Pure ablation logic: zero out scores if strictly in 'hallucination_only' mode to see text-only capability
        if mode == "hallucination_only_text":
             detector_input_scores = {k: 0.0 for k in detector_input_scores}

        if mode in ("full", "hallucination_only", "hallucination_only_text", "evidence_hallucination", "verification_hallucination", "baseline_halluc"):
            halluc_prob = self.detector.predict(query, context, answer, detector_input_scores, mode=mode, scorer=self.scorer)
        else:
            halluc_prob = 0.5 
        
        # Update metadata dynamically based on loaded detector path
        self.model_paths["hallucination"] = str(self.detector._current_path)
        
        # Fix 9: Hallucination label (Fix 5: uses dynamic threshold)
        halluc_label = "FACTUAL" if halluc_prob < self.scorer.halluc_threshold else "HALLUCINATED"

        # Fix 11 & Bug #3: Support for various baseline modes with explanations
        if mode == "baseline_standard":
            vcs, calibrated_vcs = retrieval_quality, retrieval_quality
            explanation = {"mode": mode, "vcs": vcs, "reason": "Baseline: standard retrieval quality"}
        elif mode == "baseline_evidence":
            vcs = (retrieval_quality + evidence_support) / 2
            calibrated_vcs = vcs
            explanation = {"mode": mode, "vcs": vcs, "reason": "Baseline: mean of retrieval and evidence support"}
        elif mode == "baseline_halluc":
            vcs = 1.0 - halluc_prob
            calibrated_vcs = vcs
            explanation = {"mode": mode, "vcs": vcs, "reason": "Baseline: hallucination detector probability only"}
        else:
            # Step 6: Risk Finalization (Issue 5: Uses calibrated score)
            if mode == "hallucination_only_text":
                vcs, calibrated_vcs, explanation = self.scorer.score(
                    0.0, 0.0, 0.0, halluc_prob
                )
            else:
                vcs, calibrated_vcs, explanation = self.scorer.score(
                    retrieval_quality, evidence_support, verification_score, halluc_prob
                )

        # Step 6.5: Generator Answer Grounding Check (User Fix 3)
        # Only apply grounding penalty for short extractive QA answers.
        # Long-form responses (summaries, multi-sentence) will never appear verbatim in evidence spans.
        is_grounded = False
        answer_lower = answer.lower()
        is_long_form = len(answer.strip()) > 100  # Long-form responses skip grounding check
        
        if is_long_form:
            is_grounded = True  # Skip grounding for long-form answers
        elif (not answer_lower.strip() or 
            "don't know" in answer_lower or 
            "dont know" in answer_lower or 
            "cannot" in answer_lower or 
            "not found" in answer_lower or
            "no answer" in answer_lower):
            is_grounded = True
        else:
            answer_clean = "".join([c for c in answer_lower if c.isalnum() or c.isspace()]).strip()
            for e in all_evidence:
                ev_clean = "".join([c for c in e["text"].lower() if c.isalnum() or c.isspace()])
                if answer_clean in ev_clean:
                    is_grounded = True
                    break

        if not is_grounded:
            logger.warning(f"Answer '{answer}' not found in evidence spans! Applying grounding penalty.")
            vcs *= 0.5
            calibrated_vcs *= 0.5
            if explanation is not None:
                if "vcs" in explanation:
                    explanation["vcs"] = round(vcs, 4)
                if "calibrated_vcs" in explanation:
                    explanation["calibrated_vcs"] = round(calibrated_vcs, 4)
                if "reason" in explanation:
                    explanation["reason"] += " (Penalized: Answer not found in evidence)"
                explanation["grounding_penalty_applied"] = True

        risk = self.scorer.risk_level(calibrated_vcs)

        # Step 7: Compute composite hallucination score for benchmark evaluation
        # Adaptive weights based on answer type:
        # - Short extractive QA: verification-heavy (our key contribution)
        # - Long-form summaries: contradiction-detection-heavy (NLI per-claim is noisy for long text)
        if is_long_form:
            # For long-form responses: 30% answer-context similarity, 70% verification score
            composite_halluc_score = (
                0.30 * (1.0 - answer_context_sim) +
                0.70 * (1.0 - verification_score)
            )
        else:
            # For short QA: verification-heavy (entity-aware checking is strongest)
            composite_halluc_score = (
                0.35 * (1.0 - verification_score) +   # NLI entity-aware verification (key contribution)
                0.25 * (1.0 - evidence_support) +      # Evidence coverage
                0.25 * halluc_prob +                     # Learned detector
                0.15 * (1.0 - retrieval_quality)         # Retrieval quality
            )
        composite_halluc_score = float(np.clip(composite_halluc_score, 0.0, 1.0))

        result = RAGResult(
            query=query,
            answer=answer,
            retrieved_docs=docs,
            retrieved_doc_ids=retrieved_ids,
            retrieval_scores=[float(d.score) for d in docs],
            evidence_spans=all_evidence,
            contradiction_score=round(contradiction_score, 4),
            hallucination_probability=round(halluc_prob, 4),
            hallucination_label=halluc_label,
            verification_score=round(verification_score, 4),
            vcs_score=round(vcs, 4),
            calibrated_vcs=round(calibrated_vcs, 4),
            hallucination_risk=risk,
            verification_label=verification_label,
            confidence_explanation=explanation,
            model_versions=self.model_paths,
            dataset_name=dataset_name,
            component_scores={
                "retrieval": retrieval_quality,
                "evidence": evidence_support,
                "verification": verification_score,
                "hallucination": 1.0 - halluc_prob,
                "answer_context_sim": answer_context_sim,
                "answer_evidence_sim": answer_evidence_sim
            },
            retrieval_metrics={
                "retrieval_quality": round(mean_top5_score, 4),
                "best_rank": best_rank,
                "top_doc_score": round(top_doc_score, 4),
                "mean_top5_score": round(mean_top5_score, 4)
            },
            composite_halluc_score=round(composite_halluc_score, 4)
        )
        return result

    # Fix 10: Batch Evaluation API for OOD Testing
    def evaluate_dataset(self, dataset: List[Dict], mode: str = "full", dataset_name: str = "custom") -> List[RAGResult]:
        logger.info(f"Evaluating {len(dataset)} samples from {dataset_name} in mode {mode}...")
        results = []
        for item in tqdm(dataset):
            query = item.get("question") or item.get("query") or item.get("prompt", "")
            gold_id = item.get("gold_doc_id")
            if not gold_id and "gold_doc_ids" in item and item["gold_doc_ids"]:
                gold_id = item["gold_doc_ids"][0]
            
            pre_gen = item.get("pre_generated_answer") or item.get("answer") or item.get("response")
            res = self.run(query, mode=mode, dataset_name=dataset_name, gold_doc_id=gold_id, pre_generated_answer=pre_gen)
            
            # Fix 12 & Bug #4: Populate retrieval metrics if ground truth exists
            if "gold_doc_ids" in item:
                gold_ids = set(item["gold_doc_ids"])
                ret_ids = res.retrieved_doc_ids
                res.hit_rate = 1.0 if any(rid in gold_ids for rid in ret_ids) else 0.0
                res.recall_at_k = len(set(ret_ids) & gold_ids) / len(gold_ids) if gold_ids else 0.0
                res.mrr = 0.0 # Fix 4: ensure 0 instead of None
                for i, rid in enumerate(ret_ids):
                    if rid in gold_ids:
                        res.mrr = 1.0 / (i + 1)
                        break
            
            # Ground truth for ROC (Fix 7)
            if "label" in item:
                res.answer_rank_score = (1.0 - res.hallucination_probability) # Higher is Better for correlation with Factuality
                
            results.append(res)
        return results


# ──────────────────────────────────────────────────────────────
# CLI Demo
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# Research Evaluation Utilities (ECE, Brier, ROC)
# ──────────────────────────────────────────────────────────────

def save_roc_data(results: List[RAGResult], labels: List[int], output_path: str):
    """Saves data for AUROC/PR curve plotting (Gap #2)."""
    data = []
    for res, label in zip(results, labels):
        data.append({
            "vcs": res.calibrated_vcs,
            "halluc_prob": res.hallucination_probability,
            "verification": res.verification_score,
            "evidence": res.component_scores.get("evidence", 0.0),
            "retrieval": res.component_scores.get("retrieval", 0.0),
            "ground_truth": label # 1 for correct, 0 for hallucinated
        })
    pd.DataFrame(data).to_csv(output_path, index=False)
    logger.info(f"ROC Data saved to {output_path}")

if __name__ == "__main__":
    from data.dataset_loader import DatasetLoader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="Who was the first person to walk on the moon?", help="Query to run through RAG")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save artifacts")
    parser.add_argument("--dataset", type=str, default="TruthfulQA", help="Dataset name for metadata")
    parser.add_argument("--eval_batch", action="store_true", help="Run batch evaluation on a small sample")
    args = parser.parse_args()

    # Step 1: Data Preparation
    loader = DatasetLoader()
    squad_docs  = loader.load_squad_corpus(split="validation", max_docs=500)
    fever_docs  = loader.load_fever_corpus(split="labelled_dev", max_docs=500)
    corpus = squad_docs + fever_docs

    # Step 2: Pipeline Initialization
    rag = HallucinationAwareRAG()
    rag.retriever.index(corpus)

    if args.eval_batch:
        # Research Gap #3: Small scale comparison demo
        test_data = [
            {"query": "The president of USA is Joe Biden", "label": 1},
            {"query": "The moon is made of green cheese", "label": 0},
            {"query": "Python is a programming language", "label": 1}
        ]
        results = rag.evaluate_dataset(test_data, dataset_name=args.dataset)
        
        # Save ROC Data (Gap #2)
        save_roc_data(results, [d['label'] for d in test_data], Path(args.output_dir) / "roc_analysis.csv")
        
        # Compute Calibration (Gap #1: Centralized)
        vcs_probs = np.array([r.calibrated_vcs for r in results])
        gt_labels = np.array([d['label'] for d in test_data])
        ece_val = compute_ece(vcs_probs, gt_labels)
        brier_val = compute_brier_score(vcs_probs, gt_labels)
        logger.info(f"Calibration metrics: ECE={ece_val:.4f}, Brier={brier_val:.4f}")
    else:
        # Step 3: Run Single Query
        logger.info(f"Running pipeline for query: {args.query} [{args.dataset}]")
        result = rag.run(args.query, dataset_name=args.dataset)

        # Step 4: Save Publication-Ready Artifacts (Fix 10)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "predictions.json", "w") as f:
            json.dump([result.to_dict()], f, indent=2)
        
        with open(output_dir / "evidence.json", "w") as f:
            json.dump(result.evidence_spans, f, indent=2)
        
        vcs_data = [{
            "query": result.query,
            "vcs_score": result.vcs_score,
            "calibrated_vcs": result.calibrated_vcs,
            "hallucination_prob": result.hallucination_probability,
            "verification_score": result.verification_score,
            "risk": result.hallucination_risk,
            "label": result.hallucination_label
        }]
        pd.DataFrame(vcs_data).to_csv(output_dir / "vcs_scores.csv", index=False)
        
        logger.info(f"Artifacts saved to {output_dir}/")
        print("\n" + "="*50)
        print(f"QUERY: {result.query}")
        print(f"ANSWER: {result.answer}")
        print(f"CONFIDENCE: {result.calibrated_vcs:.4f} ({result.hallucination_risk} RISK)")
        print(f"LABELS: [Halu: {result.hallucination_label}] [NLI: {result.verification_label}]")
        print(f"EXPLANATION: {result.confidence_explanation['reason']}")
        print("="*50 + "\n")
