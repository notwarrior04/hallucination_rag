import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from rag_pipeline import HallucinationAwareRAG
from data.dataset_loader import DatasetLoader
from evaluate import compute_nlp_metrics
from temperature_scaling import compute_ece
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/ablation")
    args = parser.parse_args()

    # 1. Load Dataset
    loader = DatasetLoader()
    data = loader.load_ragtruth("test")[:args.max_samples]
    corpus = loader.load_squad_corpus(max_docs=500)

    # 2. Pipeline Init
    rag = HallucinationAwareRAG()
    rag.retriever.index(corpus)

    # 3. Define Ablation Modes
    modes = [
        "full",
        "evidence_only",      # No Verification, No Detection
        "verification_only",  # No Evidence, No Detection
        "hallucination_only", # No Evidence, No Verification
        "hallucination_only_text",
        "baseline_standard"   # Simple RAG
    ]

    all_performance = []

    for mode in modes:
        logger.info(f"Running ablation mode: {mode}")
        results = rag.evaluate_dataset(data, mode=mode)
        
        # Factuality Metrics
        y_true = [1 if len(d.get("labels", [])) > 0 else 0 for d in data]
        y_prob = [r.hallucination_probability for r in results]
        
        # Calibration Metrics (Factuality)
        factuality_labels = 1 - np.array(y_true)
        correctness_probs = 1.0 - np.array(y_prob)
        ece = compute_ece(correctness_probs, factuality_labels)
        from temperature_scaling import compute_brier_score
        brier = compute_brier_score(correctness_probs, factuality_labels)
        
        try:
            auc_val = roc_auc_score(y_true, y_prob)
        except:
            auc_val = 0.5
            
        res_dict = {
            "mode": mode,
            "auc": round(auc_val, 4),
            "ece": round(ece, 4),
            "brier": round(brier, 4),
            "avg_vcs": round(np.mean([r.calibrated_vcs for r in results]), 4)
        }

        # NLP metrics skipped for RAGTruth (no gold references)
        
        all_performance.append(res_dict)

    # 4. Save and Summarize
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(all_performance)
    df.to_csv(out_dir / "ablation_results.csv", index=False)
    
    logger.info("Ablation study complete.")
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()
