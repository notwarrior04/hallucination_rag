"""
evaluate.py
===========
Evaluates the full HaRAG pipeline on:
  - In-distribution  : RAGTruth test set, FEVER dev
  - Out-of-distribution: TruthfulQA, HoVer, HaluBench
  - Ablation         : HaluEval Dialog & Summarization subsets

Metrics:
  - Accuracy, F1, AUC-ROC (hallucination detection)
  - EM, F1 (QA)
  - AUROC (confidence calibration)
  - Component-level ablation

Usage:
    python evaluate.py --split in_dist --output results/eval_results.json
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
    classification_report,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ──────────────────────────────────────────────────────────────
# QA Metrics (SQuAD-style)
# ──────────────────────────────────────────────────────────────

def normalize_answer(text: str) -> str:
    import re
    import string
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ''.join(c for c in text if c not in string.punctuation)
    return ' '.join(text.split())


def exact_match(prediction: str, ground_truths: List[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(gt) for gt in ground_truths))


def token_f1(prediction: str, ground_truths: List[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue
        prec = len(common) / len(pred_tokens) if pred_tokens else 0
        rec  = len(common) / len(gt_tokens)   if gt_tokens   else 0
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
            best_f1 = max(best_f1, f1)
    return best_f1


# ──────────────────────────────────────────────────────────────
# Hallucination Detection Metrics
# ──────────────────────────────────────────────────────────────

def hallucination_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[List[float]] = None,
) -> Dict:
    acc  = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics = {
        "accuracy":  round(acc,  4),
        "precision": round(p,    4),
        "recall":    round(r,    4),
        "f1":        round(f1,   4),
    }
    if y_scores is not None:
        try:
            auc = roc_auc_score(y_true, y_scores)
            metrics["auroc"] = round(auc, 4)
        except Exception:
            pass
    return metrics


# ──────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────

def expected_calibration_error(
    confidences: List[float],
    accuracies:  List[float],
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = [(lo <= c < hi) for c in confidences]
        if sum(mask) == 0:
            continue
        bin_conf = np.mean([c for c, m in zip(confidences, mask) if m])
        bin_acc  = np.mean([a for a, m in zip(accuracies,  mask) if m])
        ece += (sum(mask) / n) * abs(bin_conf - bin_acc)
    return round(float(ece), 4)


# ──────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────

class HaRAGEvaluator:

    def __init__(self, pipeline, loader):
        self.pipeline = pipeline
        self.loader   = loader

    # ── QA (SQuAD v2) ─────────────────────────────────────────

    def evaluate_squad(self, split: str = "validation", max_samples: int = 1000) -> Dict:
        logger.info(f"Evaluating SQuAD v2 [{split}]...")
        qa_pairs = self.loader.load_squad_qa_pairs(split=split, max_pairs=max_samples)

        em_scores, f1_scores = [], []
        halu_true, halu_pred, halu_conf = [], [], []

        for item in qa_pairs:
            result = self.pipeline.run(item["question"])
            pred   = result.answer

            if item["answerable"]:
                em_scores.append(exact_match(pred,  item["answers"]))
                f1_scores.append(token_f1(pred, item["answers"]))
                halu_true.append(0)  # should not hallucinate
            else:
                halu_true.append(1)  # model should say "I don't know"
                em_scores.append(1.0 if "don't know" in pred.lower() or "cannot" in pred.lower() else 0.0)
                f1_scores.append(0.0)

            halu_pred.append(1 if result.hallucination_risk == "HIGH" else 0)
            halu_conf.append(1 - result.confidence_score)

        metrics = {
            "dataset": "squad_v2",
            "split":   split,
            "n":       len(qa_pairs),
            "EM":      round(np.mean(em_scores),  4),
            "F1":      round(np.mean(f1_scores),  4),
            **hallucination_metrics(halu_true, halu_pred, halu_conf),
        }
        logger.info(json.dumps(metrics, indent=2))
        return metrics

    # ── FEVER ─────────────────────────────────────────────────

    def evaluate_fever(self, split: str = "labelled_dev", max_samples: int = 1000) -> Dict:
        logger.info(f"Evaluating FEVER [{split}]...")
        pairs = self.loader.load_fever_pairs(split=split, max_pairs=max_samples)

        true_labels, pred_labels = [], []
        for item in pairs:
            result = self.pipeline.run(item["claim"])
            gt_bin = 1 if item["label"] == "REFUTES" else 0
            pred_bin = 1 if result.verification_label == "REFUTED" else 0
            true_labels.append(gt_bin)
            pred_labels.append(pred_bin)

        metrics = {
            "dataset": "fever",
            "split":   split,
            "n":       len(pairs),
            **hallucination_metrics(true_labels, pred_labels),
        }
        logger.info(json.dumps(metrics, indent=2))
        return metrics

    # ── HaluEval ──────────────────────────────────────────────

    def evaluate_halueval(
        self,
        subset: str = "qa_samples",
        max_samples: int = 500,
    ) -> Dict:
        logger.info(f"Evaluating HaluEval [{subset}]...")
        samples = self.loader.load_halueval(subset=subset, max_samples=max_samples * 2)

        true_labels, pred_labels, scores = [], [], []
        for s in samples[:max_samples]:
            # Test hallucinated answer detection
            for answer, is_halu in [(s["right"], 0), (s["halluc"], 1)]:
                if not answer:
                    continue
                query = s.get("text", "")[:200]
                result = self.pipeline.run(query + " " + answer[:100])
                pred   = 1 if result.hallucination_risk == "HIGH" else 0
                true_labels.append(is_halu)
                pred_labels.append(pred)
                scores.append(1 - result.confidence_score)

        metrics = {
            "dataset": f"halueval_{subset}",
            "n":       len(true_labels),
            **hallucination_metrics(true_labels, pred_labels, scores),
        }
        logger.info(json.dumps(metrics, indent=2))
        return metrics

    # ── TruthfulQA ────────────────────────────────────────────

    def evaluate_truthfulqa(self, max_samples: int = 500) -> Dict:
        logger.info("Evaluating TruthfulQA (OOD)...")
        samples = self.loader.load_truthfulqa(max_samples=max_samples)

        em_scores, confidences = [], []
        for s in samples:
            result = self.pipeline.run(s["question"])
            em     = exact_match(result.answer, [s["best_answer"]])
            em_scores.append(em)
            confidences.append(result.confidence_score)

        ece = expected_calibration_error(confidences, em_scores)
        metrics = {
            "dataset":     "truthfulqa",
            "n":           len(samples),
            "EM":          round(np.mean(em_scores), 4),
            "mean_conf":   round(np.mean(confidences), 4),
            "ECE":         ece,
        }
        logger.info(json.dumps(metrics, indent=2))
        return metrics

    # ── HaluBench ─────────────────────────────────────────────

    def evaluate_halubench(self, max_samples: int = 500) -> Dict:
        logger.info("Evaluating HaluBench (OOD)...")
        try:
            samples = self.loader.load_halubench(max_samples=max_samples)
        except Exception as e:
            logger.warning(f"HaluBench not available: {e}")
            return {}

        true_labels, pred_labels, scores = [], [], []
        for s in samples:
            result = self.pipeline.run(s.get("question", s.get("text", "")))
            gt_bin = 1 if s.get("label", "").lower() in ("hallucinated", "false", "refuted") else 0
            pred   = 1 if result.hallucination_risk in ("HIGH", "MEDIUM") else 0
            true_labels.append(gt_bin)
            pred_labels.append(pred)
            scores.append(1 - result.confidence_score)

        metrics = {
            "dataset": "halubench",
            "n":       len(true_labels),
            **hallucination_metrics(true_labels, pred_labels, scores),
        }
        logger.info(json.dumps(metrics, indent=2))
        return metrics

    # ── Ablation ──────────────────────────────────────────────

    def ablation_study(self, halueval_samples: List[Dict], max_samples: int = 200) -> Dict:
        """
        Tests removing each component to measure its contribution.
        Variants: full | no_highlighter | no_verifier | no_scorer
        """
        logger.info("Running ablation study...")
        from rag_pipeline import HallucinationAwareRAG, HybridRetriever, Generator

        base_retriever = self.pipeline.retriever
        base_generator = self.pipeline.generator

        def run_variant(pipeline_variant, samples):
            true_l, pred_l = [], []
            for s in samples[:max_samples]:
                for answer, is_halu in [(s["right"], 0), (s["halluc"], 1)]:
                    if not answer:
                        continue
                    result = pipeline_variant.run(s.get("text", "")[:100])
                    true_l.append(is_halu)
                    pred_l.append(1 if result.hallucination_risk == "HIGH" else 0)
            return hallucination_metrics(true_l, pred_l)

        results = {}
        for variant_name, kwargs in [
            ("full",           {}),
            ("no_highlighter", {"highlighter": None}),
            ("no_verifier",    {"verifier": None}),
        ]:
            # Build variant (None components fall back to dummies in pipeline)
            try:
                variant = HallucinationAwareRAG(
                    retriever=base_retriever,
                    generator=base_generator,
                    **kwargs,
                )
                results[variant_name] = run_variant(variant, halueval_samples)
            except Exception as e:
                logger.warning(f"Ablation variant {variant_name} failed: {e}")
                results[variant_name] = {"error": str(e)}

        logger.info(json.dumps(results, indent=2))
        return results


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["in_dist", "ood", "ablation", "all"], default="all")
    parser.add_argument("--output", default="results/eval_results.json")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--corpus_size",  type=int, default=5000)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data.dataset_loader import DatasetLoader
    from rag_pipeline import HallucinationAwareRAG, HybridRetriever

    loader   = DatasetLoader()
    pipeline = HallucinationAwareRAG()

    # Index corpus
    corpus = loader.load_squad_corpus(max_docs=args.corpus_size // 2)
    corpus += loader.load_fever_corpus(max_docs=args.corpus_size // 2)
    pipeline.retriever.index(corpus)

    evaluator = HaRAGEvaluator(pipeline, loader)
    all_results = {}

    if args.split in ("in_dist", "all"):
        all_results["squad_v2"]   = evaluator.evaluate_squad(max_samples=args.max_samples)
        all_results["fever_dev"]  = evaluator.evaluate_fever(max_samples=args.max_samples)

    if args.split in ("ood", "all"):
        all_results["truthfulqa"] = evaluator.evaluate_truthfulqa(max_samples=args.max_samples)
        all_results["halubench"]  = evaluator.evaluate_halubench(max_samples=args.max_samples)

    if args.split in ("ablation", "all"):
        halu = loader.load_halueval(max_samples=args.max_samples)
        all_results["ablation"]   = evaluator.ablation_study(halu, max_samples=200)
        all_results["halueval_qa"] = evaluator.evaluate_halueval("qa_samples",      args.max_samples)
        all_results["halueval_dialog"] = evaluator.evaluate_halueval("dialogue_samples", args.max_samples // 2)
        all_results["halueval_summ"]   = evaluator.evaluate_halueval("summarization_samples", args.max_samples // 2)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
