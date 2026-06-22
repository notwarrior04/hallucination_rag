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
from temperature_scaling import TemperatureScaler, compute_ece, compute_brier_score

def fit_vcs_scaler(vcs_scores, labels):
    """
    Fits a temperature scaler specifically on VCS logits.
    """
    eps = 1e-8
    clamped = np.clip(vcs_scores, eps, 1.0 - eps)
    logits = np.log(clamped / (1.0 - clamped))
    
    # We want to calibrate confidence to factuality (1 = factual, 0 = hallucinated)
    # The temperature scaler optimizes probability of class 1.
    scaler = TemperatureScaler()
    # Fit scaler (labels are 1 for factual, 0 for hallucinated)
    scaler.fit(logits, labels)
    return scaler

def main():
    parser = argparse.ArgumentParser(description="Evaluate Calibration before and after Temperature Scaling")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples to evaluate")
    parser.add_argument("--output_file", type=str, default="evaluation/results/calibration_evaluation_results.json", help="Path to save results")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    loader = DatasetLoader()
    pipeline = HallucinationAwareRAG()

    # Load HaluEval for calibration evaluation (balanced correct/hallucinated)
    logger.info("Loading HaluEval samples for calibration evaluation...")
    qa_samples = loader.load_halueval("qa_samples", max_samples=args.max_samples)
    
    if not qa_samples:
        logger.error("Could not load HaluEval samples.")
        return

    # Build corpus
    corpus = []
    seen_contexts = set()
    for s in qa_samples:
        ctx = s.get("text", "")
        if ctx and ctx not in seen_contexts:
            seen_contexts.add(ctx)
            corpus.append({
                "doc_id": f"calib_ctx_{len(corpus)}",
                "text": ctx,
                "source": "halueval"
            })
    
    logger.info(f"Indexing corpus of {len(corpus)} documents...")
    pipeline.retriever.index(corpus)
    ctx_to_id = {c["text"]: c["doc_id"] for c in corpus}

    # Construct balanced eval suite
    eval_suite = []
    for s in qa_samples:
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
                "label": 0  # Factual
            })

        halluc_ans = s.get("halluc", "")
        if halluc_ans:
            eval_suite.append({
                "question": query,
                "pre_generated_answer": halluc_ans,
                "gold_doc_id": gold_doc_id,
                "label": 1  # Hallucinated
            })

    random.shuffle(eval_suite)
    logger.info(f"Running calibration evaluation on {len(eval_suite)} samples...")

    # Run pipeline and collect scores
    composite_scores = []
    vcs_scores_raw = []
    vcs_scores_calibrated_mismatched = []
    halluc_probs = []
    y_true = []

    for idx, item in enumerate(eval_suite):
        res = pipeline.run(
            query=item["question"],
            pre_generated_answer=item["pre_generated_answer"],
            gold_doc_id=item["gold_doc_id"],
            mode="full",
            dataset_name="Calibration"
        )
        
        composite_scores.append(res.composite_halluc_score)
        vcs_scores_raw.append(res.vcs_score)
        vcs_scores_calibrated_mismatched.append(res.calibrated_vcs)
        halluc_probs.append(res.hallucination_probability)
        y_true.append(item["label"])

        if (idx + 1) % 20 == 0 or (idx + 1) == len(eval_suite):
            logger.info(f"Evaluated {idx + 1}/{len(eval_suite)} samples...")

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    factuality_labels = 1 - y_true # 1 = factual, 0 = hallucinated
    
    composite_scores = np.array(composite_scores)
    vcs_scores_raw = np.array(vcs_scores_raw)
    vcs_scores_calibrated_mismatched = np.array(vcs_scores_calibrated_mismatched)
    halluc_probs = np.array(halluc_probs)

    # Split into dev (fit T_vcs) and test (evaluate ECE)
    n_dev = len(eval_suite) // 2
    dev_vcs_raw = vcs_scores_raw[:n_dev]
    dev_labels = factuality_labels[:n_dev]
    
    test_vcs_raw = vcs_scores_raw[n_dev:]
    test_vcs_mismatched = vcs_scores_calibrated_mismatched[n_dev:]
    test_composite = 1.0 - composite_scores[n_dev:]
    test_detector = 1.0 - halluc_probs[n_dev:]
    test_labels = factuality_labels[n_dev:]

    # Fit scaling specifically for VCS logits on the dev split
    logger.info(f"Fitting TemperatureScaler specifically on VCS logits of dev split ({n_dev} samples)...")
    vcs_scaler = fit_vcs_scaler(dev_vcs_raw, dev_labels)
    t_vcs = vcs_scaler.temperature
    logger.info(f"VCS Temperature T_vcs: {t_vcs:.4f}")

    # Scale the test VCS logits using the VCS scaler
    eps = 1e-8
    test_clamped = np.clip(test_vcs_raw, eps, 1.0 - eps)
    test_logits = np.log(test_clamped / (1.0 - test_clamped))
    test_vcs_calibrated_proper = 1.0 / (1.0 + np.exp(-test_logits / t_vcs))

    methods = {
        "Raw Detector (1-halluc_prob)": test_detector,
        "Raw VCS (uncalibrated)": test_vcs_raw,
        "Calibrated VCS (mismatched T_detector)": test_vcs_mismatched,
        "Calibrated VCS (proper T_vcs)": test_vcs_calibrated_proper,
        "Composite (1-composite_score)": test_composite,
    }

    metrics = {
        "dataset": "HaluEval (Calibration)",
        "n_samples": len(eval_suite),
        "n_test": len(test_labels),
        "detector_temperature": round(pipeline.scorer.scaler.temperature, 4),
        "vcs_temperature": round(t_vcs, 4)
    }

    print("\n" + "=" * 80)
    print("CALIBRATION PERFORMANCE SUMMARY (TEST SPLIT)")
    print("=" * 80)
    print(f"{'Method':40s} | {'ECE':10s} | {'Brier':10s}")
    print("-" * 80)

    for name, probs in methods.items():
        ece = compute_ece(probs, test_labels)
        brier = compute_brier_score(probs, test_labels)
        
        metrics[name] = {
            "ece": round(ece, 4),
            "brier": round(brier, 4)
        }
        print(f"{name:40s} | {ece:10.4f} | {brier:10.4f}")

    print("=" * 80)
    print(f"Mismatched Temperature (T_det) : {pipeline.scorer.scaler.temperature:.4f}")
    print(f"Proper Temperature (T_vcs)     : {t_vcs:.4f}")
    print("=" * 80 + "\n")

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Calibration metrics successfully saved to {args.output_file}")

if __name__ == "__main__":
    main()
