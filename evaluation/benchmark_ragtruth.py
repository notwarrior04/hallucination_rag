import os
import sys
import json
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Reconfigure stdout to use UTF-8 to handle unicode characters on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Insert root folder to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_loader import DatasetLoader
from rag_pipeline import HallucinationAwareRAG
from temperature_scaling import compute_ece, compute_brier_score


def find_optimal_threshold(scores, labels, metric_fn=f1_score):
    """Find the threshold that maximizes the given metric on a dev set."""
    best_t, best_val = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = [1 if s > t else 0 for s in scores]
        val = metric_fn(labels, preds, zero_division=0)
        if val > best_val:
            best_val = val
            best_t = t
    return best_t, best_val


def main():
    parser = argparse.ArgumentParser(description="Run RAGTruth Benchmark")
    parser.add_argument("--max_samples", type=int, default=150, help="Max samples to evaluate")
    parser.add_argument("--output_metrics", type=str, default="evaluation/results/ragtruth_metrics.json", help="Path to save metrics")
    parser.add_argument("--output_raw", type=str, default="evaluation/results/raw_predictions/ragtruth_raw_predictions.json", help="Path to save raw predictions")
    parser.add_argument("--dev_ratio", type=float, default=0.2, help="Fraction of samples used for threshold tuning")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    loader = DatasetLoader()
    pipeline = HallucinationAwareRAG()

    logger.info("Loading RAGTruth test samples...")
    raw_samples = loader.load_ragtruth(split="test", max_samples=args.max_samples * 2)
    
    if not raw_samples:
        logger.error("RAGTruth samples could not be loaded. Verify dataset is downloaded or available.")
        return

    # Build corpus of unique source_info texts
    corpus = []
    seen_contexts = set()
    for s in raw_samples:
        ctx = s.get("text", "")
        if ctx and ctx not in seen_contexts:
            seen_contexts.add(ctx)
            corpus.append({
                "doc_id": f"ragtruth_ctx_{len(corpus)}",
                "text": ctx,
                "source": "ragtruth"
            })

    logger.info(f"Indexing RAGTruth context corpus of size {len(corpus)}...")
    pipeline.retriever.index(corpus)

    # Context to doc_id mapping
    ctx_to_id = {c["text"]: c["doc_id"] for c in corpus}

    # Construct evaluation suite
    eval_suite = []
    for s in raw_samples:
        ctx = s.get("text", "")
        gold_doc_id = ctx_to_id.get(ctx)
        query = s.get("prompt", "")
        ans = s.get("response", "")
        label = s.get("label", 0)  # 1 for hallucinated, 0 for correct

        if not query or not ans or gold_doc_id is None:
            continue

        eval_suite.append({
            "question": query,
            "pre_generated_answer": ans,
            "gold_doc_id": gold_doc_id,
            "label": label
        })

    # Limit to max_samples
    eval_suite = eval_suite[:args.max_samples]
    logger.info(f"Evaluating RAGTruth benchmark with {len(eval_suite)} samples.")

    raw_predictions = []
    all_composite_scores = []
    all_halluc_probs = []
    all_labels = []
    recalls_5 = []
    recalls_10 = []
    mrrs = []

    for idx, item in enumerate(eval_suite):
        q = item["question"]
        ans = item["pre_generated_answer"]
        gold_id = item["gold_doc_id"]
        label = item["label"]

        res = pipeline.run(
            query=q,
            pre_generated_answer=ans,
            gold_doc_id=gold_id,
            mode="full",
            dataset_name="RAGTruth"
        )

        all_composite_scores.append(res.composite_halluc_score)
        all_halluc_probs.append(res.hallucination_probability)
        all_labels.append(label)

        # Compute retrieval quality
        ret_ids = [doc.base_doc_id for doc in res.retrieved_docs]
        r5 = 1.0 if (gold_id is not None and gold_id in ret_ids[:5]) else 0.0
        r10 = 1.0 if (gold_id is not None and gold_id in ret_ids[:10]) else 0.0
        
        mrr = 0.0
        if gold_id is not None:
            for rank_idx, rid in enumerate(ret_ids):
                if rid == gold_id:
                    mrr = 1.0 / (rank_idx + 1)
                    break
        
        recalls_5.append(r5)
        recalls_10.append(r10)
        mrrs.append(mrr)

        # Save raw predictions
        raw_predictions.append({
            "question": q,
            "answer": ans[:200],
            "ground_truth": label,
            "hallucination_probability": round(res.hallucination_probability, 4),
            "composite_halluc_score": round(res.composite_halluc_score, 4),
            "vcs_score": round(res.calibrated_vcs, 4),
            "verification_score": round(res.verification_score, 4),
            "component_scores": {k: round(v, 4) for k, v in res.component_scores.items()}
        })

        if (idx + 1) % 20 == 0 or (idx + 1) == len(eval_suite):
            logger.info(f"Evaluated {idx + 1}/{len(eval_suite)} samples...")

    # Split into dev (threshold tuning) and test (final evaluation)
    n_dev = max(10, int(len(eval_suite) * args.dev_ratio))
    dev_scores = all_composite_scores[:n_dev]
    dev_labels = all_labels[:n_dev]
    test_scores = all_composite_scores[n_dev:]
    test_labels = all_labels[n_dev:]

    # Find optimal threshold on dev set
    optimal_threshold, dev_f1 = find_optimal_threshold(dev_scores, dev_labels)
    logger.info(f"Optimal threshold from dev set: {optimal_threshold:.2f} (dev F1: {dev_f1:.4f})")

    # Evaluate on ALL data with optimized threshold
    y_pred_all = [1 if s > optimal_threshold else 0 for s in all_composite_scores]
    acc_all = accuracy_score(all_labels, y_pred_all)
    prec_all = precision_score(all_labels, y_pred_all, zero_division=0)
    rec_all = recall_score(all_labels, y_pred_all, zero_division=0)
    f1_all = f1_score(all_labels, y_pred_all, zero_division=0)
    
    try:
        auroc_all = roc_auc_score(all_labels, all_composite_scores)
    except Exception:
        auroc_all = 0.5

    # Test set metrics
    y_pred_test = [1 if s > optimal_threshold else 0 for s in test_scores]
    test_acc = accuracy_score(test_labels, y_pred_test)
    test_f1 = f1_score(test_labels, y_pred_test, zero_division=0)
    try:
        test_auroc = roc_auc_score(test_labels, test_scores)
    except:
        test_auroc = 0.5

    # Compute calibration metrics
    factuality_labels = 1 - np.array(all_labels)
    correctness_probs = 1.0 - np.array(all_composite_scores)
    ece = compute_ece(correctness_probs, factuality_labels)
    brier = compute_brier_score(correctness_probs, factuality_labels)

    # Aggregate retrieval quality
    mean_recall_5 = float(np.mean(recalls_5))
    mean_recall_10 = float(np.mean(recalls_10))
    mean_mrr = float(np.mean(mrrs))

    metrics = {
        "dataset": "RAGTruth",
        "n_samples": len(eval_suite),
        "n_test": len(test_labels),
        "optimal_threshold": round(optimal_threshold, 4),
        "accuracy": round(acc_all, 4),
        "precision": round(prec_all, 4),
        "recall": round(rec_all, 4),
        "f1": round(f1_all, 4),
        "auroc": round(auroc_all, 4),
        "test_accuracy": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
        "test_auroc": round(test_auroc, 4),
        "ece": round(ece, 4),
        "brier": round(brier, 4),
        "mean_recall@5": round(mean_recall_5, 4),
        "mean_recall@10": round(mean_recall_10, 4),
        "mean_mrr": round(mean_mrr, 4)
    }

    # Print summary
    print("\n" + "=" * 60)
    print("RAGTRUTH BENCHMARK RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print("=" * 60 + "\n")

    # Save metrics JSON
    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # Add prediction classification based on optimal threshold
    for item in raw_predictions:
        item["prediction"] = 1 if item["composite_halluc_score"] > optimal_threshold else 0

    # Save raw predictions JSON
    Path(args.output_raw).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_raw, "w") as f:
        json.dump(raw_predictions, f, indent=2)

    logger.info(f"Metrics saved to {args.output_metrics}")
    logger.info(f"Raw predictions saved to {args.output_raw}")

if __name__ == "__main__":
    main()
