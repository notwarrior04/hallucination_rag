import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Insert root folder to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_loader import DatasetLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def compute_bucket_metrics(samples, score_key="composite_halluc_score"):
    if not samples:
        return {"count": 0, "auroc": 0.0, "accuracy": 0.0, "f1": 0.0}
    
    labels = [s["ground_truth"] for s in samples]
    scores = [s[score_key] for s in samples]
    
    preds = [s.get("prediction", 0 if s[score_key] < 0.5 else 1) for s in samples]
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    
    try:
        if len(set(labels)) > 1:
            auroc = roc_auc_score(labels, scores)
        else:
            auroc = 0.5
    except Exception:
        auroc = 0.5
        
    return {
        "count": len(samples),
        "auroc": round(auroc, 4),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4)
    }

def build_dataset_maps():
    loader = DatasetLoader()
    
    # 1. HaluEval Maps
    logger.info("Loading HaluEval for length mapping...")
    qa = loader.load_halueval("qa_samples", max_samples=1000)
    dial = loader.load_halueval("dialogue_samples", max_samples=1000)
    summ = loader.load_halueval("summarization_samples", max_samples=1000)
    all_halueval = qa + dial + summ
    
    halueval_map = {}
    for s in all_halueval:
        q = s.get("question", "")
        # For summarization, question might be empty, fallback
        if not q:
            q = s.get("text", "")[:200]
        r = s.get("right", "")
        if r:
            halueval_map[(q, r[:200])] = r
        h = s.get("halluc", "")
        if h:
            halueval_map[(q, h[:200])] = h

    # 2. RAGTruth Maps
    logger.info("Loading RAGTruth for length mapping...")
    ragtruth_samples = loader.load_ragtruth(split="test", max_samples=2000)
    ragtruth_map = {}
    for s in ragtruth_samples:
        q = s.get("prompt", "")
        resp = s.get("response", "")
        if resp:
            # Match by question prompt and response prefix
            ragtruth_map[(q, resp[:200])] = resp
            
    return halueval_map, ragtruth_map

def analyze_dataset(predictions_path: Path, data_map, score_key="composite_halluc_score") -> dict:
    if not predictions_path.exists():
        logger.warning(f"File not found: {predictions_path}")
        return {}

    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    buckets = {
        "0-50 tokens": [],
        "50-100 tokens": [],
        "100-200 tokens": [],
        "200+ tokens": []
    }

    match_count = 0
    fallback_count = 0

    for item in data:
        q = item.get("question", "")
        ans_prefix = item.get("answer", "")
        
        # Look up full untruncated answer
        full_ans = data_map.get((q, ans_prefix))
        if full_ans:
            match_count += 1
        else:
            # Try to match just by answer prefix if question format differed slightly
            matched = False
            for (map_q, map_prefix), map_full in data_map.items():
                if map_prefix == ans_prefix:
                    full_ans = map_full
                    matched = True
                    match_count += 1
                    break
            if not matched:
                full_ans = ans_prefix
                fallback_count += 1
                
        tokens = len(full_ans.strip().split())
        
        if tokens <= 50:
            buckets["0-50 tokens"].append(item)
        elif tokens <= 100:
            buckets["50-100 tokens"].append(item)
        elif tokens <= 200:
            buckets["100-200 tokens"].append(item)
        else:
            buckets["200+ tokens"].append(item)

    logger.info(f"Analyzed {predictions_path.name}: Matched untruncated answers={match_count}, Fallback={fallback_count}")

    bucket_results = {}
    for b_name, b_samples in buckets.items():
        bucket_results[b_name] = compute_bucket_metrics(b_samples, score_key)

    return bucket_results

def main():
    raw_dir = Path("evaluation/results/raw_predictions")
    output_dir = Path("evaluation/results/error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting length bucket analysis on predictions...")

    halueval_map, ragtruth_map = build_dataset_maps()

    halueval_results = analyze_dataset(raw_dir / "halueval_raw_predictions.json", halueval_map)
    ragtruth_results = analyze_dataset(raw_dir / "ragtruth_raw_predictions.json", ragtruth_map)

    results = {
        "halueval": halueval_results,
        "ragtruth": ragtruth_results
    }

    # Save to JSON
    json_out = output_dir / "length_bucket_analysis.json"
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Length bucket analysis saved to {json_out}")

    # Generate Markdown Report
    md_report = "# Length-based Detection Performance Analysis\n\n"
    md_report += "This report analyzes how hallucination detection performance scales with generated response length (estimated in whitespace-split tokens).\n\n"
    
    for ds_name, ds_res in [("HaluEval", halueval_results), ("RAGTruth", ragtruth_results)]:
        md_report += f"## {ds_name} Length Buckets\n\n"
        md_report += "| Response Length | Sample Count | Accuracy | F1-Score | AUROC |\n"
        md_report += "| --- | --- | --- | --- | --- |\n"
        for bucket, metrics in ds_res.items():
            md_report += f"| {bucket} | {metrics['count']} | {metrics['accuracy']:.4f} | {metrics['f1']:.4f} | {metrics['auroc']:.4f} |\n"
        md_report += "\n"

    md_out = output_dir / "length_bucket_analysis.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_report)
    logger.info(f"Length bucket analysis Markdown report saved to {md_out}")

    # Print results to console
    print("\n" + "="*80)
    print("LENGTH BUCKETS ANALYSIS")
    print("="*80)
    for ds_name, ds_res in [("HaluEval", halueval_results), ("RAGTruth", ragtruth_results)]:
        print(f"\nDataset: {ds_name}")
        print("-" * 80)
        print(f"{'Length Bucket':15s} | {'Count':5s} | {'Acc':6s} | {'F1':6s} | {'AUROC':6s}")
        print("-" * 80)
        for bucket, metrics in ds_res.items():
            print(f"{bucket:15s} | {metrics['count']:5d} | {metrics['accuracy']:.4f} | {metrics['f1']:.4f} | {metrics['auroc']:.4f}")
        print("-" * 80)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
