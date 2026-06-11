"""
publication_tables.py
=====================
Aggregates results from evaluation, ablation, and calibration directories
to generate publication-ready tables (CSV/LaTeX).
"""

import os
import json
import pandas as pd
from pathlib import Path

def generate_main_results_table(eval_dir: Path):
    """Table 1: Main Results on RAGTruth, HaluEval, etc."""
    rows = []
    for f in eval_dir.glob("*_full_metrics.json"):
        with open(f, "r") as j:
            data = json.load(j)
        rows.append({
            "Dataset": data.get("dataset"),
            "Accuracy": data.get("accuracy"),
            "F1": data.get("f1"),
            "AUC": data.get("auc"),
            "ECE": data.get("ece"),
            "NLP F1": data.get("nlp_f1")
        })
    return pd.DataFrame(rows)

def generate_ablation_table(ablation_file: Path):
    """Table 2: Ablation Study Results."""
    if not ablation_file.exists(): return None
    return pd.read_csv(ablation_file)

def generate_ood_table(ood_dir: Path):
    """Table 3: OOD Generalization Results."""
    rows = []
    for f in ood_dir.glob("*_ood_metrics.json"):
        with open(f, "r") as j:
            data = json.load(j)
        rows.append({
            "Dataset": data.get("dataset"),
            "Samples": data.get("n_samples"),
            "Accuracy": data.get("accuracy"),
            "AUC": data.get("auc")
        })
    return pd.DataFrame(rows)

def main():
    root_results = Path("results")
    tables_dir = root_results / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Main Results
    df_main = generate_main_results_table(root_results / "evaluation")
    if not df_main.empty:
        df_main.to_csv(tables_dir / "table1_main_results.csv", index=False)
        print("Table 1 generated.")
        
    # 2. Ablations
    df_ablation = generate_ablation_table(root_results / "ablation" / "ablation_results.csv")
    if df_ablation is not None:
        df_ablation.to_csv(tables_dir / "table2_ablations.csv", index=False)
        print("Table 2 generated.")
        
    # 3. OOD
    df_ood = generate_ood_table(root_results / "ood")
    if not df_ood.empty:
        df_ood.to_csv(tables_dir / "table3_ood.csv", index=False)
        print("Table 3 generated.")

    print(f"\nAll publication tables saved to {tables_dir}/")

if __name__ == "__main__":
    main()
