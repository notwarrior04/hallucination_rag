import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from pathlib import Path

# Use a clean aesthetic with matplotlib defaults
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'axes.grid': True})

def plot_roc_pr_curves(data_path: str, output_dir: Path):
    """Plots ROC and PR curves for all components."""
    if not Path(data_path).exists():
        print(f"Data for ROC not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    y_true = df["ground_truth"] # 1 for Factual, 0 for Hallucinated

    # --- ROC CURVE ---
    plt.figure(figsize=(6, 5))
    
    # VCS ROC
    fpr, tpr, _ = roc_curve(y_true, df["vcs"])
    plt.plot(fpr, tpr, label=f'VCS (AUC = {auc(fpr, tpr):.3f})', lw=2, color='blue')
    
    # 1 - HallucProb ROC
    fpr, tpr, _ = roc_curve(y_true, 1.0 - df["halluc_prob"])
    plt.plot(fpr, tpr, label=f'1-Halluc (AUC = {auc(fpr, tpr):.3f})', linestyle='--', color='red')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig(output_dir / "roc_curves.png", bbox_inches='tight')
    plt.close()

    # --- PR CURVE ---
    plt.figure(figsize=(6, 5))
    precision, recall, _ = precision_recall_curve(y_true, df["vcs"])
    ap = average_precision_score(y_true, df["vcs"])
    plt.plot(recall, precision, label=f'VCS (AP = {ap:.3f})', lw=2, color='green')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (VCS)')
    plt.legend(loc='lower left')
    plt.savefig(output_dir / "pr_curve.png", bbox_inches='tight')
    plt.close()

def plot_calibration_curve(data_path: str, output_dir: Path, n_bins: int = 10):
    """Plots calibration curve (Reliability Diagram)."""
    if not Path(data_path).exists(): return
    df = pd.read_csv(data_path)
    
    y_true = df["ground_truth"]
    y_prob = df["vcs"]
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        if np.any(in_bin):
            accuracies.append(np.mean(y_true[in_bin]))
            confidences.append(np.mean(y_prob[in_bin]))
            counts.append(np.sum(in_bin))
        else:
            accuracies.append(0)
            confidences.append((bin_lower + bin_upper) / 2)
            counts.append(0)
            
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.plot(confidences, accuracies, "s-", label='VCS')
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Fraction of Positives (Accuracy)")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend(loc="lower right")
    plt.savefig(output_dir / "calibration_curve.png", bbox_inches='tight')
    plt.close()

def plot_ablation_chart(ablation_path: str, output_dir: Path):
    """Plots ablation results bar chart."""
    if not Path(ablation_path).exists(): return
    df = pd.read_csv(ablation_path)
    
    plt.figure(figsize=(8, 5))
    x = np.arange(len(df['mode']))
    width = 0.35
    
    plt.bar(x - width/2, df['nlp_f1'], width, label='NLP F1', color='skyblue')
    plt.bar(x + width/2, df['auc'], width, label='Factuality AUC', color='salmon')
    
    plt.xlabel('Ablation Mode')
    plt.ylabel('Score')
    plt.title('Ablation Study Results')
    plt.xticks(x, df['mode'], rotation=15)
    plt.legend()
    plt.savefig(output_dir / "ablation_chart.png", bbox_inches='tight')
    plt.close()

def main():
    results_dir = Path("results/evaluation")
    ablation_dir = Path("results/ablation")
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate charts
    plot_roc_pr_curves(results_dir / "RAGTruth_full_roc_data.csv", plots_dir)
    plot_calibration_curve(results_dir / "RAGTruth_full_roc_data.csv", plots_dir)
    plot_ablation_chart(ablation_dir / "ablation_results.csv", plots_dir)
    
if __name__ == "__main__":
    main()
