import os
import sys
import json
import logging
from pathlib import Path

# Reconfigure stdout to use UTF-8 to handle unicode characters on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def analyze_predictions(raw_predictions_path: Path) -> dict:
    if not raw_predictions_path.exists():
        logger.warning(f"Raw predictions file not found: {raw_predictions_path}")
        return {}

    with open(raw_predictions_path, "r") as f:
        data = json.load(f)

    tp, fp, tn, fn = [], [], [], []

    for item in data:
        gt = item.get("ground_truth")
        pred = item.get("prediction")
        
        # Binary confusion classification
        if gt == 1 and pred == 1:
            tp.append(item)
        elif gt == 0 and pred == 1:
            fp.append(item)
        elif gt == 0 and pred == 0:
            tn.append(item)
        elif gt == 1 and pred == 0:
            fn.append(item)

    total = len(data)
    tps, fps, tns, fns = len(tp), len(fp), len(tn), len(fn)

    # Classification metrics
    accuracy = (tps + tns) / total if total > 0 else 0.0
    precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
    recall = tps / (tps + fns) if (tps + fns) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Failure Analysis: categorize false positives and false negatives
    # For example, look at average VCS score, average hallucination probability, etc.
    avg_fp_prob = sum(item.get("hallucination_probability", 0) for item in fp) / max(fps, 1)
    avg_fn_prob = sum(item.get("hallucination_probability", 0) for item in fn) / max(fns, 1)

    avg_fp_vcs = sum(item.get("vcs_score", 0) for item in fp) / max(fps, 1)
    avg_fn_vcs = sum(item.get("vcs_score", 0) for item in fn) / max(fns, 1)

    return {
        "dataset_name": raw_predictions_path.stem.replace("_raw_predictions", ""),
        "total_samples": total,
        "confusion_matrix": {
            "True_Positives": tps,
            "False_Positives": fps,
            "True_Negatives": tns,
            "False_Negatives": fns
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        },
        "failure_metrics": {
            "avg_fp_hallucination_prob": round(avg_fp_prob, 4),
            "avg_fn_hallucination_prob": round(avg_fn_prob, 4),
            "avg_fp_vcs_score": round(avg_fp_vcs, 4),
            "avg_fn_vcs_score": round(avg_fn_vcs, 4)
        },
        "false_positive_examples": fp[:3],  # Select first 3 examples for error inspection
        "false_negative_examples": fn[:3]
    }

def main():
    raw_dir = Path("evaluation/results/raw_predictions")
    output_dir = Path("evaluation/results/error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting error analysis on raw predictions...")

    datasets = ["halueval", "ragtruth"]
    reports = {}

    for ds in datasets:
        raw_path = raw_dir / f"{ds}_raw_predictions.json"
        logger.info(f"Analyzing {ds} predictions...")
        analysis = analyze_predictions(raw_path)
        if analysis:
            reports[ds] = analysis
            
            # Print high level statistics
            print("\n" + "="*60)
            print(f"ERROR ANALYSIS SUMMARY: {ds.upper()}")
            print("="*60)
            print(f"Total Evaluated: {analysis['total_samples']}")
            print(f"Confusion Matrix:")
            print(f"  True Positives  (Hallucinations Detected) : {analysis['confusion_matrix']['True_Positives']}")
            print(f"  False Positives (Factual Flagged Error)   : {analysis['confusion_matrix']['False_Positives']}")
            print(f"  True Negatives  (Factual Correctly Pass)  : {analysis['confusion_matrix']['True_Negatives']}")
            print(f"  False Negatives (Hallucinations Missed)   : {analysis['confusion_matrix']['False_Negatives']}")
            print(f"Metrics:")
            print(f"  Accuracy  : {analysis['metrics']['accuracy']:.4f}")
            print(f"  F1-Score  : {analysis['metrics']['f1']:.4f}")
            print(f"Average Failure Statistics:")
            print(f"  Avg FP Hallucination Probability : {analysis['failure_metrics']['avg_fp_hallucination_prob']:.4f}")
            print(f"  Avg FN Hallucination Probability : {analysis['failure_metrics']['avg_fn_hallucination_prob']:.4f}")
            print("="*60 + "\n")

    # Save to JSON
    out_file = output_dir / "error_analysis_report.json"
    with open(out_file, "w") as f:
        json.dump(reports, f, indent=2)
    logger.info(f"Successfully generated and saved error analysis report to {out_file}")

    # Write a clean markdown file for presentation
    md_file = output_dir / "error_analysis_report.md"
    with open(md_file, "w") as f:
        f.write("# Error Analysis & Failure Case Inspection\n\n")
        f.write("Detailed failure analysis of false positives (factual content flagged as hallucination) and false negatives (hallucinations missed by the detector).\n\n")
        
        for ds, report in reports.items():
            f.write(f"## {ds.upper()} Failure Analysis\n\n")
            f.write("### Quantitative Summary\n\n")
            f.write("| Metric | Value |\n")
            f.write("| --- | --- |\n")
            f.write(f"| Accuracy | {report['metrics']['accuracy']:.4f} |\n")
            f.write(f"| Precision | {report['metrics']['precision']:.4f} |\n")
            f.write(f"| Recall | {report['metrics']['recall']:.4f} |\n")
            f.write(f"| F1-Score | {report['metrics']['f1']:.4f} |\n")
            f.write(f"| False Positives (FP) | {report['confusion_matrix']['False_Positives']} |\n")
            f.write(f"| False Negatives (FN) | {report['confusion_matrix']['False_Negatives']} |\n")
            f.write(f"| Avg FP Hallucination Prob | {report['failure_metrics']['avg_fp_hallucination_prob']:.4f} |\n")
            f.write(f"| Avg FN Hallucination Prob | {report['failure_metrics']['avg_fn_hallucination_prob']:.4f} |\n\n")
            
            f.write("### Qualitative Inspection (Failure Cases)\n\n")
            
            f.write("#### 1. False Positives (Factual Answers Flagged as Hallucination)\n")
            if report["false_positive_examples"]:
                for idx, ex in enumerate(report["false_positive_examples"]):
                    f.write(f"**Example {idx+1}**:\n")
                    f.write(f"- **Question**: {ex['question']}\n")
                    f.write(f"- **Answer**: {ex['answer']}\n")
                    f.write(f"- **Predicted Hallucination Prob**: {ex['hallucination_probability']:.4f}\n")
                    f.write(f"- **VCS Confidence Score**: {ex['vcs_score']:.4f}\n\n")
            else:
                f.write("*None observed.*\n\n")

            f.write("#### 2. False Negatives (Hallucinations Missed by Detector)\n")
            if report["false_negative_examples"]:
                for idx, ex in enumerate(report["false_negative_examples"]):
                    f.write(f"**Example {idx+1}**:\n")
                    f.write(f"- **Question**: {ex['question']}\n")
                    f.write(f"- **Answer**: {ex['answer']}\n")
                    f.write(f"- **Predicted Hallucination Prob**: {ex['hallucination_probability']:.4f}\n")
                    f.write(f"- **VCS Confidence Score**: {ex['vcs_score']:.4f}\n\n")
            else:
                f.write("*None observed.*\n\n")
            f.write("---\n\n")

    logger.info(f"Markdown error analysis report saved to {md_file}")

if __name__ == "__main__":
    main()
