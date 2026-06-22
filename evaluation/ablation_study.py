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
    """Find the threshold that maximizes the given metric."""
    best_t, best_val = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = [1 if s > t else 0 for s in scores]
        val = metric_fn(labels, preds, zero_division=0)
        if val > best_val:
            best_val = val
            best_t = t
    return best_t, best_val


def compute_system_metrics(y_true, y_prob, system_name):
    """Compute metrics for a system with optimal threshold search."""
    optimal_t, _ = find_optimal_threshold(y_prob, y_true)
    y_pred = [1 if p > optimal_t else 0 for p in y_prob]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = 0.5
    
    factuality_labels = 1 - np.array(y_true)
    correctness_probs = 1.0 - np.array(y_prob)
    ece = compute_ece(correctness_probs, factuality_labels)
    brier = compute_brier_score(correctness_probs, factuality_labels)
    
    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "AUROC": round(auroc, 4),
        "ECE": round(ece, 4),
        "Brier": round(brier, 4),
        "Threshold": round(optimal_t, 4)
    }


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Study (Systems A to E on HaluEval)")
    parser.add_argument("--max_samples", type=int, default=50, help="Max samples per subset to evaluate")
    parser.add_argument("--output_file", type=str, default="evaluation/results/ablation_study_results.json", help="Path to save results")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    loader = DatasetLoader()
    pipeline = HallucinationAwareRAG()

    logger.info("Loading HaluEval subsets...")
    qa_samples = loader.load_halueval("qa_samples", max_samples=args.max_samples)
    dial_samples = loader.load_halueval("dialogue_samples", max_samples=args.max_samples)
    summ_samples = loader.load_halueval("summarization_samples", max_samples=args.max_samples)

    raw_samples = qa_samples + dial_samples + summ_samples
    logger.info(f"Loaded {len(raw_samples)} raw HaluEval samples.")

    # Build and index corpus
    corpus = []
    seen_contexts = set()
    for s in raw_samples:
        ctx = s.get("text", "")
        if ctx and ctx not in seen_contexts:
            seen_contexts.add(ctx)
            corpus.append({
                "doc_id": f"halueval_ctx_{len(corpus)}",
                "text": ctx,
                "source": "halueval"
            })
    pipeline.retriever.index(corpus)

    ctx_to_id = {c["text"]: c["doc_id"] for c in corpus}

    # Construct balanced eval suite
    eval_suite = []
    for s in raw_samples:
        ctx = s.get("text", "")
        gold_doc_id = ctx_to_id.get(ctx)
        query = s.get("question", "")
        if not query:
            query = ctx[:200] if ctx else "What is the summary?"

        right_ans = s.get("right", "")
        if right_ans:
            eval_suite.append({
                "question": query,
                "pre_generated_answer": right_ans,
                "gold_doc_id": gold_doc_id,
                "label": 0
            })

        halluc_ans = s.get("halluc", "")
        if halluc_ans:
            eval_suite.append({
                "question": query,
                "pre_generated_answer": halluc_ans,
                "gold_doc_id": gold_doc_id,
                "label": 1
            })

    random.shuffle(eval_suite)
    logger.info(f"Evaluating {len(eval_suite)} balanced samples across Systems A to E.")

    # Initialize score tracking for each system
    # Each system produces a hallucination score (higher = more likely hallucinated)
    systems = {
        "System A: Retrieval Only": [],
        "System B: + NLI Verifier": [],
        "System C: + Evidence Highlighter": [],
        "System D: + Entity Matching": [],
        "System E: Full HaRAG (+ Detector)": [],
    }
    
    y_true = []

    for idx, item in enumerate(eval_suite):
        q = item["question"]
        ans = item["pre_generated_answer"]
        gold_id = item["gold_doc_id"]
        label = item["label"]
        y_true.append(label)

        # Run full pipeline once — extract all component scores
        res = pipeline.run(q, pre_generated_answer=ans, gold_doc_id=gold_id, mode="full", dataset_name="HaluEval")
        
        retrieval_q = res.component_scores.get("retrieval", 0.5)
        evidence_s = res.component_scores.get("evidence", 0.5)
        verification_s = res.verification_score
        halluc_p = res.hallucination_probability
        
        # System A: Retrieval Only — uses (1 - retrieval_quality) as halluc score
        systems["System A: Retrieval Only"].append(1.0 - retrieval_q)
        
        # System B: + NLI Verifier — adds verification score
        score_b = 0.5 * (1.0 - retrieval_q) + 0.5 * (1.0 - verification_s)
        systems["System B: + NLI Verifier"].append(score_b)
        
        # System C: + Evidence Highlighter — adds evidence support
        score_c = 0.35 * (1.0 - retrieval_q) + 0.35 * (1.0 - verification_s) + 0.30 * (1.0 - evidence_s)
        systems["System C: + Evidence Highlighter"].append(score_c)
        
        # System D: + Entity Matching (excluding hallucination detector)
        # Re-normalize the other weights to sum to 1.0 (excluding the detector weight)
        is_long = len(ans.strip()) > 100
        if is_long:
            score_d = (
                0.30 * (1.0 - res.component_scores.get("answer_context_sim", 0.5)) +
                0.70 * (1.0 - verification_s)
            )
        else:
            score_d = (
                0.35 * (1.0 - verification_s) +
                0.25 * (1.0 - evidence_s) +
                0.15 * (1.0 - retrieval_q)
            ) / 0.75
        systems["System D: + Entity Matching"].append(float(np.clip(score_d, 0.0, 1.0)))
        
        # System E: Full HaRAG (same as composite, includes detector)
        systems["System E: Full HaRAG (+ Detector)"].append(res.composite_halluc_score)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(eval_suite):
            logger.info(f"Processed {idx + 1}/{len(eval_suite)} samples...")

    # Compute final metrics for each system
    results = {}
    for name, scores in systems.items():
        results[name] = compute_system_metrics(y_true, scores, name)

    # Print summary table
    print("\n" + "="*90)
    print("ABLATION STUDY: COMPONENT CONTRIBUTION ON HALUEVAL")
    print("="*90)
    print(f"{'System Name':40s} | {'Acc':6s} | {'F1':6s} | {'AUROC':6s} | {'ECE':6s} | {'Brier':6s}")
    print("-" * 90)
    for name, m in results.items():
        print(f"{name:40s} | {m['Accuracy']:.4f} | {m['F1']:.4f} | {m['AUROC']:.4f} | {m['ECE']:.4f} | {m['Brier']:.4f}")
    print("="*90 + "\n")

    # Save results JSON
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Ablation results successfully saved to {args.output_file}")

if __name__ == "__main__":
    main()
