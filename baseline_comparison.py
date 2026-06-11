"""
baseline_comparison.py
======================
Statistical comparison between Standard RAG and HaRAG variants.
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from rag_pipeline import HallucinationAwareRAG
from data.dataset_loader import DatasetLoader
from evaluate import compute_nlp_metrics
from sklearn.metrics import roc_auc_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    loader = DatasetLoader()
    data = loader.load_ragtruth("test")[:args.max_samples]
    corpus = loader.load_squad_corpus(max_docs=500)

    rag = HallucinationAwareRAG()
    rag.retriever.index(corpus)

    comparisons = [
        ("Standard RAG", "baseline_standard"),
        ("RAG + Evidence", "evidence_only"),
        ("RAG + Verifier", "verification_only"),
        ("RAG + Detector", "hallucination_only"),
        ("Full HaRAG", "full")
    ]

    results_summary = []

    for name, mode in comparisons:
        print(f"Evaluating {name}...")
        results = rag.evaluate_dataset(data, mode=mode)
        
        # Factuality Metrics
        y_true = [1 if len(d.get("labels", [])) > 0 else 0 for d in data]
        y_prob = [r.hallucination_probability for r in results]
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.5
            
        res_row = {
            "Method": name,
            "Factuality AUC": round(auc, 4),
            "Avg Confidence": round(pd.Series([r.calibrated_vcs for r in results]).mean(), 4)
        }

        # RAGTruth has no gold references. Skip NLP metrics.
            
        results_summary.append(res_row)

    df = pd.DataFrame(results_summary)
    print("\n" + df.to_string(index=False))
    
    output_dir = Path("results/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "baseline_comparison.csv", index=False)

if __name__ == "__main__":
    main()
