"""
train_models.py
===============
Trains / fine-tunes all three HaRAG components with advanced features:

  1. Evidence Highlighter  — Cross-encoder fine-tuned on SQuAD v2 + FEVER + HoVer
  2. Contradiction Verifier — NLI model fine-tuned on FEVER + HoVer + RAGTruth contradiction samples
  3. Hallucination Detector  — Redesigned as binary classifier (BCEWithLogitsLoss) using
                              concatenated features (Retrieval Score, Evidence Score, NLI Score,
                              Answer Text, Retrieved Context) on HaluEval + RAGTruth

Includes Temperature Scaling, Calibration Plots (Reliability Diagram), ROC curve, Validation Curves,
Ablation studies, Baselines, OOD Evaluation Hooks, and Model Card generation.
"""

import argparse
import json
import logging
import os
import random
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
)
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)

import matplotlib
matplotlib.use('Agg')  # Headless mode for server / terminal running
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

FEVER_LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 2, "NOT ENOUGH INFO": 1}


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────
# 1. Evidence Highlighter Training (Cross-Encoder)
# ──────────────────────────────────────────────────────────────

class CombinedEvidenceDataset(Dataset):
    """
    Combined dataset for training Evidence Highlighter.
    Positive: (query/claim, gold passage/sentence) -> 1.0
    Negative: (query/claim, random non-evidence sentence) -> 0.0
    """

    def __init__(self, squad_pairs: List[Dict], fever_pairs: List[Dict], hover_samples: List[Dict], negatives_per_pos: int = 3):
        self.examples = []

        # 1. Process SQuAD
        squad_passages = [p["context"] for p in squad_pairs]
        for item in squad_pairs:
            if not item["answerable"]:
                continue
            q_a = f"{item['question']} [SEP] {item['answers'][0]}"
            pos = item["context"]
            self.examples.append(InputExample(texts=[q_a, pos], label=1.0))
            squad_pool = [p for p in squad_passages if p != pos]
            negs = random.sample(squad_pool, min(negatives_per_pos, len(squad_pool)))
            for neg in negs:
                self.examples.append(InputExample(texts=[q_a, neg], label=0.0))

        # 2. Process FEVER
        fever_evidence_pool = []
        for item in fever_pairs:
            fever_evidence_pool.extend(item.get("evidence", []))
        fever_evidence_pool = [e for e in fever_evidence_pool if e]

        for item in fever_pairs:
            claim = item["claim"]
            evs = item.get("evidence", [])
            for ev in evs:
                if ev:
                    self.examples.append(InputExample(texts=[claim, ev], label=1.0))
                    fever_pool = [e for e in fever_evidence_pool if e != ev]
                    negs = random.sample(fever_pool, min(negatives_per_pos, len(fever_pool)))
                    for neg in negs:
                        self.examples.append(InputExample(texts=[claim, neg], label=0.0))

        # 3. Process HoVer
        hover_evidence_pool = []
        for item in hover_samples:
            sents = [s.strip() for s in item.get("evidence", "").split("\n") if len(s.strip()) > 10]
            hover_evidence_pool.extend(sents)
        hover_evidence_pool = [e for e in hover_evidence_pool if e]

        for item in hover_samples:
            claim = item["claim"]
            sents = [s.strip() for s in item.get("evidence", "").split("\n") if len(s.strip()) > 10]
            for sent in sents:
                self.examples.append(InputExample(texts=[claim, sent], label=1.0))
                hover_pool = [e for e in hover_evidence_pool if e != sent]
                negs = random.sample(hover_pool, min(negatives_per_pos, len(hover_pool)))
                for neg in negs:
                    self.examples.append(InputExample(texts=[claim, neg], label=0.0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train_evidence_highlighter(
    squad_pairs: List[Dict],
    fever_pairs: List[Dict],
    hover_samples: List[Dict],
    output_dir: str,
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 3,
    batch_size: int = 8,
):
    logger.info("Training Evidence Highlighter (Cross-Encoder)...")
    output_path = Path(output_dir) / "evidence_highlighter"
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = CombinedEvidenceDataset(squad_pairs, fever_pairs, hover_samples, negatives_per_pos=3)
    if len(dataset) == 0:
        raise RuntimeError("Dataset empty. Check dataset loading.")
    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_examples = [dataset[i] for i in train_ds.indices]
    val_examples   = [dataset[i] for i in val_ds.indices]
    
    train_pairs = [(e.texts[0], e.texts[1]) for e in train_examples]
    train_labels = [e.label for e in train_examples]
    val_pairs   = [(e.texts[0], e.texts[1]) for e in val_examples]
    val_labels  = [e.label for e in val_examples]

    model = CrossEncoder(base_model, num_labels=1, device=DEVICE)

    evaluator = CECorrelationEvaluator.from_input_examples(
        [InputExample(texts=list(p), label=l) for p, l in zip(val_pairs[:500], val_labels[:500])],
        name="combined_val"
    )

    train_examples = [
        InputExample(texts=list(p), label=l)
        for p, l in zip(train_pairs, train_labels)
    ]

    model.fit(
        train_dataloader=DataLoader(train_examples, shuffle=True, batch_size=batch_size),
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=500,
        output_path=str(output_path),
        save_best_model=True,
        use_amp=torch.cuda.is_available(),
    )
    logger.info(f"Evidence Highlighter saved to {output_path}")
    return str(output_path)


# ──────────────────────────────────────────────────────────────
# 2. Contradiction Verifier Training (NLI)
# ──────────────────────────────────────────────────────────────

class CombinedNLIDataset(Dataset):
    """
    Unified NLI Dataset.
    Labels: ENTAILMENT (0), NEUTRAL (1), CONTRADICTION (2)
    """

    def __init__(self, fever_pairs: List[Dict], hover_samples: List[Dict], ragtruth_samples: List[Dict], tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []

        # 1. FEVER
        for item in fever_pairs:
            if item.get("claim") and item.get("label") in FEVER_LABEL_MAP:
                ev = item.get("evidence", [])
                if ev and isinstance(ev[0], list):
                    premise = " ".join([str(s) for s in ev[0] if s])
                else:
                    premise = str(ev[0]) if ev else item["claim"]

                self.items.append({
                    "premise": premise,
                    "hypothesis": item["claim"],
                    "label": FEVER_LABEL_MAP[item["label"]]
                })

        # 2. HoVer (label 0 = Supported -> 0, label 1 = Not-Supported -> 2)
        for item in hover_samples:
            if item.get("claim") and item.get("evidence"):
                self.items.append({
                    "premise": item["evidence"],
                    "hypothesis": item["claim"],
                    "label": 0 if item["label"] == 0 else 2
                })

        # 3. RAGTruth Contradiction/Neutral Samples
        for item in ragtruth_samples:
            premise = item.get("text", "")
            hypothesis = item.get("response", "")
            if premise and hypothesis:
                labels_list = item.get("labels", [])
                if not labels_list:
                    label = 0  # Entailment
                elif any("Conflict" in lbl.get("label_type", "") for lbl in labels_list):
                    label = 2  # Contradiction
                else:
                    label = 1  # Neutral (Baseless Info)

                self.items.append({
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": label
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        enc = self.tokenizer(
            item["premise"],
            item["hypothesis"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }


def train_contradiction_verifier(
    fever_pairs: List[Dict],
    hover_samples: List[Dict],
    ragtruth_samples: List[Dict],
    output_dir: str,
    base_model: str = "roberta-base",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
):
    logger.info("Training Contradiction Verifier (NLI on FEVER+HoVer+RAGTruth)...")
    output_path = Path(output_dir) / "contradiction_verifier"
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from {base_model}: {e}. Trying use_fast=False.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        except Exception as e2:
            logger.warning(f"Failed to load with use_fast=False: {e2}. Falling back to roberta-base.")
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    dataset   = CombinedNLIDataset(fever_pairs, hover_samples, ragtruth_samples, tokenizer)
    if len(dataset) == 0:
        raise RuntimeError("Dataset empty. Check dataset loading.")
    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)
    model.config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    model.config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    device = DEVICE
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_acc = -1.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch  = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds  = logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total   += len(batch["labels"])

        val_acc = correct / max(total, 1)
        avg_loss = total_loss / max(len(train_loader), 1)
        logger.info(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            logger.info(f"  ✓ Saved best model (acc={val_acc:.4f})")

    logger.info(f"Contradiction Verifier saved to {output_path}")
    return str(output_path)


# ──────────────────────────────────────────────────────────────
# Predictors for Scorer Feature Extraction
# ──────────────────────────────────────────────────────────────

class EvidenceHighlighterPredictor:
    def __init__(self, model_path_or_name: str):
        self.cross_encoder = CrossEncoder(model_path_or_name, device=DEVICE)

    def highlight(self, query: str, answer: str, context: str, top_k: int = 2) -> List[Dict]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', context.strip())
        sentences = [s for s in sentences if len(s) > 10]
        if not sentences:
            return []

        windows = []
        for i in range(len(sentences)):
            for w in range(1, 4):
                end = min(i + w, len(sentences))
                span = " ".join(sentences[i:end])
                windows.append((span, i, end - 1))

        query_answer = f"{query} [SEP] {answer}"
        pairs = [(query_answer, span) for span, _, _ in windows]

        scores = self.cross_encoder.predict(pairs)
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
                    "score": float(scores[idx])
                })
        return sorted(evidence, key=lambda x: x["score"], reverse=True)


class ContradictionModelPredictor:
    def __init__(self, model_path_or_name: str, device):
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {model_path_or_name}: {e}. Trying use_fast=False.")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_fast=False)
            except Exception as e2:
                logger.warning(f"Failed to load with use_fast=False: {e2}. Falling back to roberta-base.")
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path_or_name)
        self.model.to(device)
        self.model.eval()

        # Determine contradiction index dynamically (fallback index 2: entailment=0, neutral=1, contradiction=2)
        self.contradiction_idx = 2
        if hasattr(self.model.config, "label2id") and self.model.config.label2id:
            label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
            for k, idx in label2id.items():
                if "contradict" in k:
                    self.contradiction_idx = idx
                    logger.info(f"Contradiction index detected dynamically: {self.contradiction_idx} (label: '{k}')")
                    break

    def verify(self, answer: str, evidence_spans: List[Dict]) -> Tuple[float, str]:
        if not evidence_spans:
            return 0.0, "NEI"

        contradiction_scores = []
        for span in evidence_spans:
            with torch.no_grad():
                enc = self.tokenizer(
                    span["text"],
                    answer,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                ).to(self.device)
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                # Label indices: dynamically mapped to model configuration
                contradiction_scores.append(probs[self.contradiction_idx])

        avg_contra = float(np.mean(contradiction_scores))
        if avg_contra > 0.4:
            label = "REFUTED"
        else:
            label = "NEI"

        return avg_contra, label


def extract_features(sample, highlighter, verifier, bi_encoder):
    query = sample.get("query", sample.get("question", ""))
    context = sample.get("context", sample.get("text", ""))
    answer = sample.get("answer", sample.get("response", ""))

    if not query:
        query = context[:200] if context else "dummy query"
    if not context:
        context = "dummy context"
    if not answer:
        answer = "dummy answer"

    # 1. Retrieval Score
    q_emb = bi_encoder.encode(query, convert_to_tensor=True)
    d_emb = bi_encoder.encode(context[:512], convert_to_tensor=True)
    retrieval_score = float(torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), d_emb.unsqueeze(0)).cpu().numpy()[0])
    retrieval_score = float(np.clip(retrieval_score, 0.0, 1.0))

    # 2. Evidence Score (sigmoid scaling on raw logits)
    spans = highlighter.highlight(query, answer, context, top_k=2)
    if spans:
        raw_score = np.mean([s["score"] for s in spans])
        evidence_score = float(1.0 / (1.0 + np.exp(-raw_score)))
    else:
        evidence_score = 0.0

    # 3. NLI Score
    contradiction_score, _ = verifier.verify(answer, spans)
    nli_score = float(np.clip(contradiction_score, 0.0, 1.0))

    return {
        "retrieval_score": retrieval_score,
        "evidence_score": evidence_score,
        "nli_score": nli_score,
    }


# ──────────────────────────────────────────────────────────────
# 3. Hallucination Detector (Combined Classifier & Dataset)
# ──────────────────────────────────────────────────────────────

class CombinedHallucinationDataset(Dataset):
    """
    Combined HaluEval + RAGTruth Hallucination Dataset.
    Precomputes Retrieval, Evidence, and NLI features.
    """

    def __init__(self, halueval_samples: List[Dict], ragtruth_samples: List[Dict], tokenizer, highlighter, verifier, bi_encoder, max_length: int = 384, cache_path: str = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []
        
        # Check cache if provided
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading precomputed features from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                self.items = cached_data["items"]
                self.precomputed_features = cached_data["features"]
            return

        # 1. HaluEval samples
        logger.info("Processing HaluEval samples for CombinedHallucinationDataset...")
        for s in halueval_samples:
            context = s.get("text", "")
            query = s.get("question", "")
            if not query:
                query = context[:200]

            right_ans = s.get("right", "")
            if right_ans:
                self.items.append({
                    "id": s["id"] + "_right",
                    "query": query,
                    "context": context,
                    "answer": right_ans,
                    "label": 0.0,  # correct
                    "source": "halueval"
                })
            halluc_ans = s.get("halluc", "")
            if halluc_ans:
                self.items.append({
                    "id": s["id"] + "_halluc",
                    "query": query,
                    "context": context,
                    "answer": halluc_ans,
                    "label": 1.0,  # hallucinated
                    "source": "halueval"
                })

        # 2. RAGTruth samples
        logger.info("Processing RAGTruth samples for CombinedHallucinationDataset...")
        for s in ragtruth_samples:
            context = s.get("text", "")
            query = s.get("prompt", "")
            if not query:
                query = context[:200]
            answer = s.get("response", "")
            if not answer:
                continue

            label = float(s.get("label") if "label" in s else (1.0 if len(s.get("labels", [])) > 0 else 0.0))

            self.items.append({
                "id": s["id"],
                "query": query,
                "context": context,
                "answer": answer,
                "label": label,
                "source": "ragtruth"
            })

        # Precompute features (Optimized version)
        logger.info(f"Extracting scores/features for {len(self.items)} samples...")
        
        # Batch encode all unique queries and contexts for speed
        all_queries = list(set(item["query"] for item in self.items))
        all_contexts = list(set(item["context"][:512] for item in self.items))
        
        logger.info(f"Encoding {len(all_queries)} unique queries and {len(all_contexts)} unique contexts...")
        q_embeddings = {txt: emb for txt, emb in zip(all_queries, bi_encoder.encode(all_queries, batch_size=64, show_progress_bar=True, convert_to_tensor=True))}
        c_embeddings = {txt: emb for txt, emb in zip(all_contexts, bi_encoder.encode(all_contexts, batch_size=64, show_progress_bar=True, convert_to_tensor=True))}
        
        self.precomputed_features = []
        for item in tqdm(self.items, desc="Precomputing hallucination features"):
            # 1. Retrieval Score (Precomputed)
            q_emb = q_embeddings[item["query"]]
            d_emb = c_embeddings[item["context"][:512]]
            retrieval_score = float(torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), d_emb.unsqueeze(0)).cpu().numpy()[0])
            retrieval_score = float(np.clip(retrieval_score, 0.0, 1.0))
            
            # 2. Evidence & NLI Scores (These still require highlighter/verifier calls)
            query = item["query"]
            context = item["context"]
            answer = item["answer"]
            
            # Evidence Score
            spans = highlighter.highlight(query, answer, context, top_k=2)
            if spans:
                raw_score = np.mean([s["score"] for s in spans])
                evidence_score = float(1.0 / (1.0 + np.exp(-raw_score)))
            else:
                evidence_score = 0.0
                
            # NLI Score
            contradiction_score, _ = verifier.verify(answer, spans)
            nli_score = float(np.clip(contradiction_score, 0.0, 1.0))
            
            self.precomputed_features.append({
                "retrieval_score": retrieval_score,
                "evidence_score": evidence_score,
                "nli_score": nli_score,
            })
            
        # Optional: Save to cache
        if cache_path:
            logger.info(f"Saving precomputed features to cache: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump({"items": self.items, "features": self.precomputed_features}, f)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        feats = self.precomputed_features[idx]

        input_text = f"Context: {item['context'][:512]} \n\n Answer: {item['answer'][:256]}"
        enc = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "retrieval_score": torch.tensor(feats["retrieval_score"], dtype=torch.float),
            "evidence_score": torch.tensor(feats["evidence_score"], dtype=torch.float),
            "nli_score": torch.tensor(feats["nli_score"], dtype=torch.float),
            "labels": torch.tensor(item["label"], dtype=torch.float)
        }


class CombinedHallucinationModel(nn.Module):
    """
    Combined Classifier model.
    Concatenates the Text Encoder's CLS embedding with custom numerical features
    depending on the ablation type.
    """

    def __init__(self, base_model: str = "roberta-base", ablation_type: str = "full"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.ablation_type = ablation_type

        if ablation_type == "hallucination_only_text":
            self.numerical_dim = 0
        elif ablation_type in ("evidence_only", "evidence_hallucination"):
            self.numerical_dim = 2  # retrieval, evidence
        elif ablation_type in ("verification_only", "verification_hallucination"):
            self.numerical_dim = 2  # retrieval, nli
        elif ablation_type == "hallucination_only":
            self.numerical_dim = 1  # retrieval only
        elif ablation_type == "evidence_verification":
            self.numerical_dim = 2  # evidence, nli
        elif ablation_type == "full":
            self.numerical_dim = 3  # retrieval, evidence, nli
        else:
            self.numerical_dim = 3

        self.classifier = nn.Sequential(
            nn.Linear(hidden + self.numerical_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask, retrieval_scores, evidence_scores, nli_scores, labels=None):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  # CLS token

        if self.ablation_type in ("evidence_only", "evidence_hallucination"):
            numerical_features = torch.stack([retrieval_scores, evidence_scores], dim=1)
        elif self.ablation_type in ("verification_only", "verification_hallucination"):
            numerical_features = torch.stack([retrieval_scores, nli_scores], dim=1)
        elif self.ablation_type == "hallucination_only":
            numerical_features = torch.stack([retrieval_scores], dim=1)
        elif self.ablation_type == "evidence_verification":
            numerical_features = torch.stack([evidence_scores, nli_scores], dim=1)
        elif self.ablation_type == "hallucination_only_text":
            numerical_features = None
        else:  # full
            numerical_features = torch.stack([retrieval_scores, evidence_scores, nli_scores], dim=1)

        if numerical_features is None:
            combined = pooled
        else:
            combined = torch.cat([pooled, numerical_features], dim=1)
        logits = self.classifier(combined).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# ──────────────────────────────────────────────────────────────
# Plotting Helpers
# ──────────────────────────────────────────────────────────────

def plot_reliability_diagram(probs_before, probs_after, labels, save_path):
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(6, 6))

    try:
        prob_true_before, prob_pred_before = calibration_curve(labels, probs_before, n_bins=10)
        plt.plot(prob_pred_before, prob_true_before, marker='o', linewidth=1, label='Before Calibration')
    except Exception as e:
        logger.warning(f"Failed to plot before calibration curve: {e}")

    try:
        prob_true_after, prob_pred_after = calibration_curve(labels, probs_after, n_bins=10)
        plt.plot(prob_pred_after, prob_true_after, marker='s', linewidth=1, label='After Calibration')
    except Exception as e:
        logger.warning(f"Failed to plot after calibration curve: {e}")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(probs, labels, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_validation_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], 'bo-', label='Training Loss')
    ax1.plot(epochs, history["val_loss"], 'ro-', label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # F1 & Accuracy curves
    ax2.plot(epochs, history["val_f1"], 'go-', label='Val F1')
    ax2.plot(epochs, history["val_accuracy"], 'yo-', label='Val Accuracy')
    ax2.set_title('Validation F1 & Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ──────────────────────────────────────────────────────────────
# OOD Evaluation Hooks & Model Card
# ──────────────────────────────────────────────────────────────

def evaluate_ood(model, tokenizer, device, dataset_name, loader, highlighter, verifier, bi_encoder):
    logger.info(f"Running OOD Evaluation on {dataset_name}...")
    if dataset_name == "truthfulqa":
        samples = loader.load_truthfulqa(max_samples=200)
        items = []
        for s in samples:
            items.append({
                "query": s["question"],
                "context": "",
                "answer": s["best_answer"],
                "label": 0.0  # Truthful answers
            })
    elif dataset_name == "hover":
        samples = loader.load_hover(split="test", max_samples=200)
        items = []
        for s in samples:
            items.append({
                "query": s["claim"],
                "context": s["evidence"],
                "answer": "",
                "label": 1.0 if str(s.get("label", "")).upper() in ("1", "1.0", "NOT_SUPPORTED", "REFUTED", "FAIL") else 0.0
            })
    elif dataset_name == "halubench":
        samples = loader.load_halubench(max_samples=200)
        items = []
        for s in samples:
            items.append({
                "query": s.get("question", ""),
                "context": s.get("text", ""),
                "answer": s.get("answer", ""),
                "label": float(item_label) if isinstance((item_label := s.get("label", 0)), (int, float)) else (1.0 if item_label == "hallucinated" else 0.0)
            })

    if not items:
        logger.warning(f"OOD dataset {dataset_name} is empty. Skipping evaluation.")
        return {}

    preds, targets, scores = [], [], []
    model.eval()
    with torch.no_grad():
        for item in items:
            feats = extract_features(item, highlighter, verifier, bi_encoder)

            input_text = f"Context: {item['context'][:512]} \n\n Answer: {item['answer'][:256]}"
            enc = tokenizer(
                input_text,
                truncation=True,
                max_length=384,
                padding="max_length",
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            retrieval_score = torch.tensor([feats["retrieval_score"]], dtype=torch.float).to(device)
            evidence_score = torch.tensor([feats["evidence_score"]], dtype=torch.float).to(device)
            nli_score = torch.tensor([feats["nli_score"]], dtype=torch.float).to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                retrieval_scores=retrieval_score,
                evidence_scores=evidence_score,
                nli_scores=nli_score
            )
            logit = out["logits"].item()
            prob = 1 / (1 + np.exp(-logit))
            pred = 1 if logit >= 0.0 else 0

            preds.append(pred)
            targets.append(int(item["label"]))
            scores.append(prob)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, zero_division=0)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)

    from temperature_scaling import compute_ece, compute_brier_score
    ece = compute_ece(np.array(scores), np.array(targets))
    brier = compute_brier_score(np.array(scores), np.array(targets))

    metrics = {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "ece": round(ece, 4),
        "brier_score": round(brier, 4)
    }
    logger.info(f"OOD {dataset_name} metrics: {metrics}")
    return metrics


def save_model_card(output_dir, training_args, metrics, ablation_type):
    readme_content = f"""# Model Card: Combined Hallucination Detector ({ablation_type})

## Model Architecture
- **Base Encoder**: {training_args.get('base_model', 'roberta-base')}
- **Classification Head**: Multi-Layer Perceptron (Concatenated CLS + Numerical Features)
- **Ablation Config**: `{ablation_type}`
- **Features Used**:
"""
    if ablation_type == "full":
        readme_content += "  - Retrieval Score (Bi-Encoder similarity between Query and Context)\n  - Evidence Score (Relevance of Query+Answer to Context Sentences via Evidence Highlighter)\n  - NLI Score (Contradiction risk of Answer against Evidence Spans via Contradiction Verifier)\n"
    elif "evidence" in ablation_type:
        readme_content += "  - Retrieval Score\n  - Evidence Score\n"
    elif "verification" in ablation_type:
        readme_content += "  - Retrieval Score\n  - NLI Score\n"
    elif ablation_type == "hallucination_only_text":
        readme_content += "  - Text Encoder Only (No Numerical Features)\n"
    elif "hallucination" in ablation_type:
        readme_content += "  - Retrieval Score\n"
    readme_content += f"""
## Training Settings
- **Epochs**: {training_args['epochs']}
- **Batch Size**: {training_args['batch_size']}
- **Learning Rate**: {training_args['lr']}
- **Seed**: {training_args['seed']}
- **Dataset Sizes**:
  - Train: {training_args['dataset_sizes']['train']}
  - Validation: {training_args['dataset_sizes']['val']}

## Evaluation Metrics (Validation Set)
- **Best F1 Score**: {metrics['best_f1']:.4f}
- **Accuracy**: {metrics['val_accuracy']:.4f}
- **Precision**: {metrics['val_precision']:.4f}
- **Recall**: {metrics['val_recall']:.4f}
- **ROC-AUC**: {metrics['val_auroc']:.4f}
- **ECE (Expected Calibration Error)**: {metrics['ece_after']:.4f}
- **Brier Score**: {metrics['brier_score_after']:.4f}
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)


# ──────────────────────────────────────────────────────────────
# Training Wrapper
# ──────────────────────────────────────────────────────────────

def run_training_for_model(
    train_ds,
    val_ds,
    tokenizer,
    ablation_type,
    output_subdir_name,
    args,
    seed=42,
    resume_checkpoint=None,
    loader=None,
    highlighter=None,
    verifier=None,
    bi_encoder=None,
):
    logger.info(f"=== Starting training for: {output_subdir_name} (Ablation: {ablation_type}) ===")
    if len(train_ds) == 0:
        raise RuntimeError("Dataset empty. Check dataset loading.")
    set_seed(seed)

    checkpoint_dir = Path(args.output_dir) / output_subdir_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = DEVICE
    model = CombinedHallucinationModel(base_model=args.base_model, ablation_type=ablation_type)

    if resume_checkpoint:
        resume_path = Path(resume_checkpoint) / "model.pt"
        if resume_path.exists():
            logger.info(f"Resuming model weights from {resume_path}")
            model.load_state_dict(torch.load(resume_path, map_location=device))

    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_accuracy": []}
    best_f1 = -1.0
    best_metrics = {
        "best_f1": 0.0,
        "val_accuracy": 0.0,
        "val_precision": 0.0,
        "val_recall": 0.0,
        "val_auroc": 0.0,
        "best_val_loss": float("inf")
    }

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            retrieval_scores = batch["retrieval_score"].to(device)
            evidence_scores = batch["evidence_score"].to(device)
            nli_scores = batch["nli_score"].to(device)
            labels = batch["labels"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                retrieval_scores=retrieval_scores,
                evidence_scores=evidence_scores,
                nli_scores=nli_scores,
                labels=labels
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_probs = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                retrieval_scores = batch["retrieval_score"].to(device)
                evidence_scores = batch["evidence_score"].to(device)
                nli_scores = batch["nli_score"].to(device)
                labels = batch["labels"].to(device)

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    retrieval_scores=retrieval_scores,
                    evidence_scores=evidence_scores,
                    nli_scores=nli_scores,
                    labels=labels
                )
                val_loss += out["loss"].item()
                logits = out["logits"].cpu().numpy()

                probs = 1.0 / (1.0 + np.exp(-logits))
                preds = (logits >= 0.0).astype(int)

                val_preds.extend(preds.tolist())
                val_probs.extend(probs.tolist())
                val_labels_list.extend(labels.cpu().numpy().tolist())

        avg_train_loss = total_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        val_acc = accuracy_score(val_labels_list, val_preds)
        val_prec = precision_score(val_labels_list, val_preds, zero_division=0)
        val_rec = recall_score(val_labels_list, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels_list, val_preds, zero_division=0)
        try:
            val_auc = roc_auc_score(val_labels_list, val_probs)
        except ValueError:
            val_auc = 0.5

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
            f"acc={val_acc:.4f} | f1={val_f1:.4f} | auc={val_auc:.4f}"
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)
        history["val_accuracy"].append(val_acc)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = {
                "best_f1": val_f1,
                "val_accuracy": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_auroc": val_auc,
                "best_val_loss": avg_val_loss
            }
            torch.save(model.state_dict(), str(checkpoint_dir / "model.pt"))
            tokenizer.save_pretrained(str(checkpoint_dir))
            logger.info(f"  ✓ Saved best model (F1={val_f1:.4f})")

    # Calibration & Metrics Export
    logger.info("Running Temperature Scaling Calibration...")
    best_model_path = checkpoint_dir / "model.pt"
    if best_model_path.exists():
        logger.info(f"Loading best model weights from {best_model_path} for calibration...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()
    val_logits_list = []
    val_labels_list = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            retrieval_scores = batch["retrieval_score"].to(device)
            evidence_scores = batch["evidence_score"].to(device)
            nli_scores = batch["nli_score"].to(device)
            labels = batch["labels"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                retrieval_scores=retrieval_scores,
                evidence_scores=evidence_scores,
                nli_scores=nli_scores
            )
            val_logits_list.extend(out["logits"].cpu().numpy().tolist())
            val_labels_list.extend(labels.cpu().numpy().tolist())

    val_logits_np = np.array(val_logits_list)
    val_labels_np = np.array(val_labels_list)

    from temperature_scaling import TemperatureScaler, compute_ece, compute_brier_score
    scaler = TemperatureScaler()
    scaler.fit(val_logits_np, val_labels_np)

    probs_before = 1.0 / (1.0 + np.exp(-val_logits_np))
    probs_after = 1.0 / (1.0 + np.exp(-scaler.scale(val_logits_np)))

    ece_before = compute_ece(probs_before, val_labels_np)
    ece_after = compute_ece(probs_after, val_labels_np)
    brier_before = compute_brier_score(probs_before, val_labels_np)
    brier_after = compute_brier_score(probs_after, val_labels_np)

    logger.info(f"ECE: Before={ece_before:.4f}, After={ece_after:.4f}")
    logger.info(f"Brier: Before={brier_before:.4f}, After={brier_after:.4f}")

    with open(checkpoint_dir / "temperature.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(checkpoint_dir / "temperature.json", "w") as f:
        json.dump({"temperature": float(scaler.temperature)}, f)

    calibration_data = {
        "temperature": scaler.temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "brier_before": brier_before,
        "brier_after": brier_after
    }
    with open(checkpoint_dir / "calibration.json", "w") as f:
        json.dump(calibration_data, f, indent=2)

    final_metrics = {
        "best_f1": best_metrics["best_f1"],
        "best_val_loss": best_metrics["best_val_loss"],
        "val_accuracy": best_metrics["val_accuracy"],
        "val_precision": best_metrics["val_precision"],
        "val_recall": best_metrics["val_recall"],
        "val_auroc": best_metrics["val_auroc"],
        "ece_after": ece_after,
        "brier_score_after": brier_after,
    }
    with open(checkpoint_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    config_data = {
        "base_model": args.base_model,
        "ablation_type": ablation_type,
        "numerical_dim": model.numerical_dim,
        "temperature": scaler.temperature
    }
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    training_args_data = {
        "base_model": args.base_model,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_train": args.max_train,
        "dataset_sizes": {
            "train": len(train_ds),
            "val": len(val_ds)
        }
    }
    with open(checkpoint_dir / "training_args.json", "w") as f:
        json.dump(training_args_data, f, indent=2)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    is_main_model = (output_subdir_name in ("confidence_scorer", "hallucination_detector_full", "hallucination_detector_halueval_ragtruth"))

    if is_main_model:
        plot_reliability_diagram(probs_before, probs_after, val_labels_np, results_dir / "reliability_diagram.png")
        plot_reliability_diagram(probs_before, probs_after, val_labels_np, checkpoint_dir / "reliability_diagram.png")

        plot_roc_curve(probs_after, val_labels_np, results_dir / "roc_curve.png")
        plot_roc_curve(probs_after, val_labels_np, checkpoint_dir / "roc_curve.png")

        plot_validation_curves(history, results_dir / "training_curve.png")
        plot_validation_curves(history, checkpoint_dir / "training_curve.png")

        with open(results_dir / "metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
        with open(results_dir / "calibration.json", "w") as f:
            json.dump(calibration_data, f, indent=2)
    else:
        plot_reliability_diagram(probs_before, probs_after, val_labels_np, checkpoint_dir / "reliability_diagram.png")
        plot_roc_curve(probs_after, val_labels_np, checkpoint_dir / "roc_curve.png")
        plot_validation_curves(history, checkpoint_dir / "training_curve.png")

    if loader and highlighter and verifier and bi_encoder:
        logger.info("Running OOD evaluations...")
        ood_results = {}
        for ood_name in ("truthfulqa", "hover", "halubench"):
            try:
                ood_metrics = evaluate_ood(model, tokenizer, device, ood_name, loader, highlighter, verifier, bi_encoder)
                ood_results[ood_name] = ood_metrics
            except Exception as e:
                logger.error(f"Failed OOD evaluation for {ood_name}: {e}", exc_info=True)

        with open(checkpoint_dir / "ood_metrics.json", "w") as f:
            json.dump(ood_results, f, indent=2)
        if is_main_model:
            with open(results_dir / "ood_metrics.json", "w") as f:
                json.dump(ood_results, f, indent=2)

    save_model_card(checkpoint_dir, training_args_data, final_metrics, ablation_type)
    logger.info(f"=== Completed training for: {output_subdir_name} ===\n")
    return final_metrics


# ──────────────────────────────────────────────────────────────
# Main Runner
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train HaRAG components")
    parser.add_argument("--component", choices=["highlighter", "verifier", "scorer", "all"], default="all")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str,   default="./checkpoints")
    parser.add_argument("--max_train",  type=int,   default=20000)
    parser.add_argument("--base_model", type=str,   default="roberta-base")
    parser.add_argument("--ablation_type", type=str, default="full",
                        choices=["evidence_only", "verification_only", "hallucination_only",
                                 "hallucination_only_text", "evidence_verification", 
                                 "evidence_hallucination", "verification_hallucination", "full"])
    parser.add_argument("--run_all_ablations", action="store_true",
                        help="Runs training for all ablation variants and baseline comparisons in a loop.")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Resume model weights from checkpoint directory.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Resolve loader
    sys.path.insert(0, str(Path(__file__).parent))
    from data.dataset_loader import DatasetLoader
    loader = DatasetLoader()

    # 1. Evidence Highlighter
    if args.component in ("highlighter", "all"):
        squad = loader.load_squad_qa_pairs(split="train", max_pairs=args.max_train)
        fever = loader.load_fever_pairs(split="train", max_pairs=args.max_train // 2)
        hover = loader.load_hover(split="train", max_samples=args.max_train // 2)
        train_evidence_highlighter(squad, fever, hover, args.output_dir, epochs=args.epochs, batch_size=args.batch_size)

    # 2. Contradiction Verifier
    if args.component in ("verifier", "all"):
        fever = loader.load_fever_pairs(split="train", max_pairs=args.max_train)
        hover = loader.load_hover(split="train", max_samples=args.max_train // 2)
        ragtruth = loader.load_ragtruth(split="train", max_samples=args.max_train // 2)
        train_contradiction_verifier(fever, hover, ragtruth, args.output_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # 3. Hallucination Scorer
    if args.component in ("scorer", "all"):
        halu = loader.load_halueval(subset="qa_samples", max_samples=args.max_train // 2)
        halu += loader.load_halueval(subset="dialogue_samples", max_samples=args.max_train // 4)
        halu += loader.load_halueval(subset="summarization_samples", max_samples=args.max_train // 4)

        ragtruth = loader.load_ragtruth(split="train", max_samples=args.max_train)

        logger.info("Loading helper models for scorer feature extraction...")
        bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

        highlighter_path = Path(args.output_dir) / "evidence_highlighter"
        if (highlighter_path / "config.json").exists():
            highlighter = EvidenceHighlighterPredictor(str(highlighter_path))
        else:
            logger.warning("No custom evidence highlighter found. Using base model.")
            highlighter = EvidenceHighlighterPredictor("cross-encoder/ms-marco-MiniLM-L-6-v2")

        verifier_path = Path(args.output_dir) / "contradiction_verifier"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (verifier_path / "config.json").exists():
            verifier = ContradictionModelPredictor(str(verifier_path), device)
        else:
            raise RuntimeError(
                "Custom contradiction verifier not found. Train verifier first."
            )

        try:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {args.base_model}: {e}. Trying use_fast=False.")
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
            except Exception as e2:
                logger.warning(f"Failed to load with use_fast=False: {e2}. Falling back to roberta-base.")
                tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        model_name_slug = args.base_model.replace("/", "_").replace("-", "_")
        cache_file = Path(args.output_dir) / f"feature_cache_{model_name_slug}.pkl"
        dataset = CombinedHallucinationDataset(
            halueval_samples=halu,
            ragtruth_samples=ragtruth,
            tokenizer=tokenizer,
            highlighter=highlighter,
            verifier=verifier,
            bi_encoder=bi_encoder,
            max_length=384,
            cache_path=str(cache_file)
        )

        num_samples = len(dataset)
        indices = list(range(num_samples))
        random.seed(args.seed)
        random.shuffle(indices)
        split_idx = int(0.9 * num_samples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        if args.run_all_ablations:
            runs = [
                ("evidence_only", "combined", "hallucination_detector_evidence_only"),
                ("verification_only", "combined", "hallucination_detector_verification_only"),
                ("hallucination_only", "combined", "hallucination_detector_hallucination_only"),
                ("hallucination_only_text", "combined", "hallucination_detector_hallucination_only_text"),
                ("evidence_verification", "combined", "hallucination_detector_evidence_verification"),
                ("evidence_hallucination", "combined", "hallucination_detector_evidence_hallucination"),
                ("verification_hallucination", "combined", "hallucination_detector_verification_hallucination"),
                ("full", "combined", "hallucination_detector_full"),
                ("full", "halueval_only", "hallucination_detector_halueval_only"),
                ("full", "combined", "hallucination_detector_halueval_ragtruth"),
            ]

            for ab_type, ds_type, subdir in runs:
                if ds_type == "combined":
                    cur_train = torch.utils.data.Subset(dataset, train_indices)
                    cur_val = torch.utils.data.Subset(dataset, val_indices)
                else:
                    he_train_ind = [i for i in train_indices if dataset.items[i].get("source") == "halueval"]
                    he_val_ind = [i for i in val_indices if dataset.items[i].get("source") == "halueval"]
                    cur_train = torch.utils.data.Subset(dataset, he_train_ind)
                    cur_val = torch.utils.data.Subset(dataset, he_val_ind)

                run_training_for_model(
                    train_ds=cur_train,
                    val_ds=cur_val,
                    tokenizer=tokenizer,
                    ablation_type=ab_type,
                    output_subdir_name=subdir,
                    args=args,
                    seed=args.seed,
                    resume_checkpoint=args.resume_checkpoint,
                    loader=loader,
                    highlighter=highlighter,
                    verifier=verifier,
                    bi_encoder=bi_encoder,
                )

            logger.info("Running Statistical Significance Testing on trained ablation models...")
            orig_argv = sys.argv.copy()
            try:
                import significance_tests
                sys.argv = [
                    "significance_tests.py",
                    "--output_dir", args.output_dir,
                    "--base_model", args.base_model,
                    "--max_train", str(args.max_train),
                    "--seed", str(args.seed),
                ]
                significance_tests.main()
            except Exception as e:
                logger.error(f"Failed to run statistical significance testing: {e}")
            finally:
                sys.argv = orig_argv
        else:
            train_ds = torch.utils.data.Subset(dataset, train_indices)
            val_ds = torch.utils.data.Subset(dataset, val_indices)

            run_training_for_model(
                train_ds=train_ds,
                val_ds=val_ds,
                tokenizer=tokenizer,
                ablation_type=args.ablation_type,
                output_subdir_name="confidence_scorer",
                args=args,
                seed=args.seed,
                resume_checkpoint=args.resume_checkpoint,
                loader=loader,
                highlighter=highlighter,
                verifier=verifier,
                bi_encoder=bi_encoder,
            )

    logger.info("All selected components trained successfully!")


if __name__ == "__main__":
    main()
