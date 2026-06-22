import os
import sys
import json
import random
import logging
import argparse
import numpy as np
from pathlib import Path

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
from evaluation.metrics import exact_match_score, fuzzy_exact_match_score, token_f1_score
from temperature_scaling import compute_ece, compute_brier_score

def main():
    parser = argparse.ArgumentParser(description="Run TruthfulQA Benchmark")
    parser.add_argument("--max_samples", type=int, default=50, help="Max samples to evaluate")
    parser.add_argument("--output_metrics", type=str, default="evaluation/results/truthfulqa_metrics.json", help="Path to save metrics")
    parser.add_argument("--output_raw", type=str, default="evaluation/results/raw_predictions/truthfulqa_raw_predictions.json", help="Path to save raw predictions")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    loader = DatasetLoader()
    pipeline = HallucinationAwareRAG()

    logger.info("Loading generic SQuAD corpus for retriever index...")
    squad_corpus = loader.load_squad_corpus(split="validation", max_docs=500)
    pipeline.retriever.index(squad_corpus)

    logger.info("Loading TruthfulQA samples...")
    raw_samples = loader.load_truthfulqa(max_samples=args.max_samples)
    
    if not raw_samples:
        logger.error("TruthfulQA samples could not be loaded. Skipping.")
        return

    logger.info(f"Evaluating TruthfulQA benchmark with {len(raw_samples)} samples.")

    raw_predictions = []
    em_scores = []
    fuzzy_em_scores = []
    f1_scores = []
    y_prob = []
    y_pred = []
    confidences = []

    for idx, item in enumerate(raw_samples):
        q = item["question"]
        best_ans = item["best_answer"]

        # Run pipeline in full mode to generate an answer
        res = pipeline.run(
            query=q,
            mode="full",
            dataset_name="TruthfulQA"
        )

        generated_ans = res.answer
        prob = res.hallucination_probability
        pred = 1 if prob > 0.5 else 0

        # Compute EM and F1 against best answer
        em = exact_match_score(generated_ans, best_ans)
        fuzzy_em = fuzzy_exact_match_score(generated_ans, best_ans)
        f1 = token_f1_score(generated_ans, best_ans)

        em_scores.append(em)
        fuzzy_em_scores.append(fuzzy_em)
        f1_scores.append(f1)
        y_prob.append(prob)
        y_pred.append(pred)
        confidences.append(res.calibrated_vcs)

        # Ground truth correctness: 1 if generated answer is wrong (EM == 0), 0 if correct
        gt_correctness = 1.0 - em

        raw_predictions.append({
            "question": q,
            "answer": generated_ans,
            "ground_truth": int(gt_correctness),
            "prediction": pred,
            "hallucination_probability": round(prob, 4),
            "vcs_score": round(res.calibrated_vcs, 4),
            "best_answer": best_ans
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(raw_samples):
            logger.info(f"Evaluated {idx + 1}/{len(raw_samples)} samples...")

    # Calculate metrics
    mean_em = float(np.mean(em_scores))
    mean_fuzzy_em = float(np.mean(fuzzy_em_scores))
    mean_f1 = float(np.mean(f1_scores))
    mean_conf = float(np.mean(confidences))

    # ECE and Brier score comparing pipeline's confidence vs actual answer correctness (EM)
    # correctness_probs = confidence (higher confidence = more correct)
    # correctness_labels = EM (1 if correct answer, 0 if wrong)
    correctness_probs = np.array(confidences)
    correctness_labels = np.array(em_scores)
    ece = compute_ece(correctness_probs, correctness_labels)
    brier = compute_brier_score(correctness_probs, correctness_labels)

    metrics = {
        "dataset": "TruthfulQA",
        "n_samples": len(raw_samples),
        "mean_em": round(mean_em, 4),
        "mean_fuzzy_em": round(mean_fuzzy_em, 4),
        "mean_f1": round(mean_f1, 4),
        "mean_confidence": round(mean_conf, 4),
        "ece": round(ece, 4),
        "brier": round(brier, 4)
    }

    # Print summary
    print("\n" + "=" * 60)
    print("TRUTHFULQA BENCHMARK RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:15s}: {v}")
    print("=" * 60 + "\n")

    # Save metrics JSON
    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save raw predictions JSON
    Path(args.output_raw).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_raw, "w") as f:
        json.dump(raw_predictions, f, indent=2)

    logger.info(f"Metrics saved to {args.output_metrics}")
    logger.info(f"Raw predictions saved to {args.output_raw}")

if __name__ == "__main__":
    main()
