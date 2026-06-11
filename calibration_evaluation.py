"""
calibration_evaluation.py
=========================
Evaluates the impact of temperature scaling on confidence calibration.
Computes ECE and Brier score before and after calibration.
"""

import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rag_pipeline import HallucinationAwareRAG
from data.dataset_loader import DatasetLoader
from temperature_scaling import compute_ece, compute_brier_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="RAGTruth")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    # 1. Load Data
    loader = DatasetLoader()
    if args.dataset == "RAGTruth":
        data = loader.load_ragtruth("test")[:args.max_samples]
        y_true = [1 if len(d.get("labels", [])) == 0 else 0 for d in data] # 1 for Factual
    else:
        # Generic handling
        data = loader.load_halueval()[:args.max_samples]
        y_true = [1 - int(d.get("label", 0)) for d in data]

    # 2. Pipeline Init
    rag = HallucinationAwareRAG()
    corpus = loader.load_squad_corpus(max_docs=500)
    rag.retriever.index(corpus)

    # 3. Predict
    logger.info(f"Running calibration evaluation on {len(data)} samples...")
    results = rag.evaluate_dataset(data)
    
    # 4. Compare Uncalibrated vs Calibrated
    # Calibrated Probs (after T scaling)
    probs_calibrated = np.array([r.calibrated_vcs for r in results])
    
    # Uncalibrated Probs (raw weighted/meta-model output before T scaling)
    probs_uncalibrated = np.array([r.vcs_score for r in results])

    # 5. Metrics
    metrics = {
        "uncalibrated": {
            "ece": round(compute_ece(probs_uncalibrated, y_true), 4),
            "brier": round(compute_brier_score(probs_uncalibrated, y_true), 4)
        },
        "calibrated": {
            "ece": round(compute_ece(probs_calibrated, y_true), 4),
            "brier": round(compute_brier_score(probs_calibrated, y_true), 4),
            "temperature": round(rag.scorer.scaler.temperature, 4)
        }
    }

    # 6. Save
    output_dir = Path("results/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{args.dataset}_calibration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Calibration Evaluation for {args.dataset}:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
