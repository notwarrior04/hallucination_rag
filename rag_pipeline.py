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
        # Query-only highlighting to avoid confirmation bias (Ultimate Fix #4)
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
        # Check if local path exists and contains config.json if it is a directory
        is_valid_local = Path(model_name).exists() and (not Path(model_name).is_dir() or (Path(model_name) / "config.json").exists())
        if not is_valid_local:
            logger.warning(f"Trained verifier not found or invalid at {model_name}. Using base model.")
            model_name = "facebook/bart-large-mnli"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.is_pipeline = True
            self.nli = pipeline("zero-shot-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
        else:
            logger.info(f"Loading contradiction verifier: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.is_pipeline = False
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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

    def _nli_predict(self, premise: str, hypothesis: str) -> Dict:
        key = f"{premise[:80]}|||{hypothesis[:80]}"
        if key in self._cache:
            return self._cache[key]

        if self.is_pipeline:
            result = self.nli(
                premise,
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template="{}",
            )
            label_scores = dict(zip(result["labels"], result["scores"]))
            out = {
                "entailment": label_scores.get("entailment", 0.0),
                "neutral":    label_scores.get("neutral",    0.0),
                "contradiction": label_scores.get("contradiction", 0.0),
            }
        else:
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

    def verify(
        self,
        answer: str,
        evidence_spans: List[Dict],
        return_entailment: bool = False
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

        contradiction_scores = []
        entailment_scores   = []

        for span in evidence_spans:
            scores = self._nli_predict(
                premise=span["text"],
                hypothesis=answer
            )
            contradiction_scores.append(scores["contradiction"])
            entailment_scores.append(scores["entailment"])

        avg_contradiction = float(np.mean(contradiction_scores))
        avg_entailment    = float(np.mean(entailment_scores))

        if avg_entailment > 0.5:
            label = "SUPPORTED"
        elif avg_contradiction > 0.4:
            label = "REFUTED"
        else:
            label = "NEI"

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
        
        tokenizer = AutoTokenizer.from_pretrained(path_str)
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
                    chunked_docs.append({
                        "doc_id": f"{doc['doc_id']}__chunk_{i}",
                        "base_doc_id": doc["doc_id"],
                        "chunk_id": f"chunk_{i}",
                        "text": chunk,
                        "source": doc.get("source", ""),
                        "title": doc.get("title", "")
                    })
                    chunk_lengths.append(len(chunk.split()))
            
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
                self.corpus.append({
                    "doc_id": doc["doc_id"],
                    "base_doc_id": doc["doc_id"],
                    "chunk_id": 0,
                    "text": doc["text"],
                    "source": doc.get("source", ""),
                    "title": doc.get("title", "")
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

    def retrieve(self, query: str, top_k: int = 20) -> Tuple[List[RetrievedDocument], np.ndarray]:
        if self.corpus_embeddings is None:
            raise RuntimeError("Call .index() before .retrieve()")

        q_emb = self.bi_encoder.encode(query, convert_to_tensor=True)
        dense_scores = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), self.corpus_embeddings
        ).cpu().numpy()

        # Fuse scores
        if self._bm25 is not None:
            bm25_scores = np.array(self._bm25.get_scores(clean_for_bm25(query)))
            
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
                
                fused_k = top_k
                rerank_k = top_k
        else:
            fused = dense_scores
            if self.fusion_method == "rrf":
                fused_k = max(150, top_k)
                rerank_k = max(75, top_k)
            else:
                fused_k = top_k
                rerank_k = top_k

        fused_sorted_indices = np.argsort(fused)[::-1]
        fused_candidates = fused_sorted_indices[:fused_k]
        rerank_candidates = fused_candidates[:rerank_k]
        
        # Then rerank top candidates
        pairs = [
            (query, self.corpus[idx]["text"])
            for idx in rerank_candidates
        ]

        rerank_scores = self.cross_encoder.predict(pairs)

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
        prompt = (
            f"Answer the question based only on the provided context. "
            f"If the answer cannot be found, say 'I don't know'.\n\n"
            f"Context: {context[:1024]}\n\nQuestion: {query}\n\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
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
        
        self.retriever   = retriever   or HybridRetriever()
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

    def run(self, query: str, top_k: int = 5, mode: str = "full", dataset_name: str = "custom", gold_doc_id: Optional[str] = None) -> RAGResult:
        # Step 1: Retrieve a larger pool to allow deduplication to top_k unique documents
        retrieval_limit = max(50, top_k * 4)
        chunks, top_dense_scores = self.retriever.retrieve(query, top_k=retrieval_limit)
        
        # Deduplicate chunks to base documents
        seen = set()
        dedup_docs = []
        for d in chunks:
            if d.base_doc_id not in seen:
                seen.add(d.base_doc_id)
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
        context = "\n\n".join([d.text for d in docs])
        with torch.no_grad():
            q_emb = self.retriever.bi_encoder.encode(query, convert_to_tensor=True)
            c_emb = self.retriever.bi_encoder.encode(context[:512], convert_to_tensor=True)
            retrieval_quality = float(torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), c_emb.unsqueeze(0)).cpu().numpy()[0])
            retrieval_quality = float(np.clip(retrieval_quality, 0.0, 1.0))

        # Step 2: Generate
        answer = self.generator.generate(query, context)

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
        # Fix 3: Evidence score normalization (matching train_models.py:582)
        raw_evidence_mean = float(np.mean([e["score"] for e in all_evidence])) if all_evidence else 0.0
        evidence_support = float(1.0 / (1.0 + np.exp(-raw_evidence_mean))) if all_evidence else 0.0
        
        # Blend evidence support with retrieval quality (User Fix 2)
        evidence_support = 0.7 * evidence_support + 0.3 * retrieval_quality

        # Step 4: Contradiction Verification
        if mode in ("full", "verification_only", "hallucination_only", "hallucination_only_text", "verification_hallucination", "evidence_verification", "baseline_halluc"):
            contradiction_score, entailment_score, verification_label = self.verifier.verify(answer, all_evidence, return_entailment=True)
        else:
            contradiction_score, entailment_score, verification_label = 0.0, 0.0, "NEI"
        
        # Use absolute NLI entailment difference to handle NEI vs SUPPORT (User Fix 1 revised)
        verification_score = max(0.0, entailment_score - contradiction_score)

        # Step 4.5: Answer Abstention (Ultimate Fix #8)
        # Force "I don't know" if the evidence support or verification score is too low
        if mode in ("full", "verification_only", "evidence_verification") and (verification_score < 0.35 or evidence_support < 0.3):
            answer_lower = answer.lower()
            if not ("don't know" in answer_lower or "dont know" in answer_lower or "cannot" in answer_lower):
                logger.warning(f"Abstaining from answer: verification_score ({verification_score:.4f}) or evidence_support ({evidence_support:.4f}) is too low.")
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
        is_grounded = False
        answer_lower = answer.lower()
        if (not answer_lower.strip() or 
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
                "evidence": evidence_support, # Issue 7: Sigmoid version for ROC
                "verification": verification_score,
                "hallucination": 1.0 - halluc_prob
            },
            retrieval_metrics={
                "retrieval_quality": round(mean_top5_score, 4),
                "best_rank": best_rank,
                "top_doc_score": round(top_doc_score, 4),
                "mean_top5_score": round(mean_top5_score, 4)
            }
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
            res = self.run(query, mode=mode, dataset_name=dataset_name, gold_doc_id=gold_id)
            
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
