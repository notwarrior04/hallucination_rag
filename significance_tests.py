import argparse
import json
import logging
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def bootstrap_accuracy_difference(y_true, preds_a, preds_b, n_bootstraps=2000, ci=0.95):
    """
    Computes bootstrap confidence interval for the accuracy difference: Acc(A) - Acc(B).
    """
    y_true = np.array(y_true)
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    n_samples = len(y_true)
    
    diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        acc_a = accuracy_score(y_true[indices], preds_a[indices])
        acc_b = accuracy_score(y_true[indices], preds_b[indices])
        diffs.append(acc_a - acc_b)
        
    diffs = np.sort(diffs)
    alpha = 1.0 - ci
    lower_idx = int(n_bootstraps * (alpha / 2))
    upper_idx = int(n_bootstraps * (1.0 - alpha / 2))
    return float(diffs[lower_idx]), float(diffs[upper_idx])

def bootstrap_f1_difference(y_true, preds_a, preds_b, n_bootstraps=2000, ci=0.95):
    """
    Computes bootstrap confidence interval for the F1 score difference: F1(A) - F1(B).
    """
    y_true = np.array(y_true)
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    n_samples = len(y_true)
    
    diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        f1_a = f1_score(y_true[indices], preds_a[indices], zero_division=0)
        f1_b = f1_score(y_true[indices], preds_b[indices], zero_division=0)
        diffs.append(f1_a - f1_b)
        
    diffs = np.sort(diffs)
    alpha = 1.0 - ci
    lower_idx = int(n_bootstraps * (alpha / 2))
    upper_idx = int(n_bootstraps * (1.0 - alpha / 2))
    return float(diffs[lower_idx]), float(diffs[upper_idx])

def mcnemar_test(preds_a, preds_b, y_true):
    """
    Computes McNemar's test p-value.
    preds_a: predictions of model A (Full HaRAG)
    preds_b: predictions of model B (Ablation model)
    y_true: ground truth labels
    """
    y_true = np.array(y_true)
    correct_a = (np.array(preds_a) == y_true).astype(int)
    correct_b = (np.array(preds_b) == y_true).astype(int)
    
    # discordant pairs
    b = int(np.sum((correct_a == 1) & (correct_b == 0)))
    c = int(np.sum((correct_a == 0) & (correct_b == 1)))
    
    total_discordant = b + c
    if total_discordant == 0:
        return 0.0, 1.0
        
    if total_discordant < 25:
        # Use exact binomial test (McNemar's exact test)
        from scipy.stats import binomtest
        result = binomtest(
            min(b, c),
            n=total_discordant,
            p=0.5,
            alternative="two-sided"
        )
        pval = result.pvalue
        return float(total_discordant), float(pval)
    else:
        # Use chi-squared test with Edwards continuity correction
        chi2 = ((abs(b - c) - 1.0) ** 2) / total_discordant
        pval = stats.chi2.sf(chi2, 1)
        return float(chi2), float(pval)

def main():
    parser = argparse.ArgumentParser(description="Statistical Significance Testing for HaRAG Ablation Models")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Path to checkpoints directory")
    parser.add_argument("--base_model", type=str, default="roberta-base", help="Base model architecture name")
    parser.add_argument("--max_train", type=int, default=20000, help="Max train/dataset samples limit")
    parser.add_argument("--seed", type=int, default=42, help="Seed value used for training/splits split")
    args = parser.parse_args()

    # Import workspace dependencies
    sys.path.insert(0, str(Path(__file__).parent))
    from train_models import (
        CombinedHallucinationDataset,
        CombinedHallucinationModel,
        EvidenceHighlighterPredictor,
        ContradictionModelPredictor,
    )
    from data.dataset_loader import DatasetLoader
    from sentence_transformers import SentenceTransformer

    # 1. Initialize Loader & Load Datasets
    logger.info("Loading validation dataset...")
    loader = DatasetLoader()
    
    # Mirror exactly the training dataset compilation
    halu = []
    halu += loader.load_halueval(
        subset="qa_samples",
        max_samples=args.max_train // 2
    )
    halu += loader.load_halueval(
        subset="dialogue_samples",
        max_samples=args.max_train // 4
    )
    halu += loader.load_halueval(
        subset="summarization_samples",
        max_samples=args.max_train // 4
    )
    
    ragtruth = loader.load_ragtruth(split="train", max_samples=args.max_train)

    # Load Helper Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    highlighter_path = Path(args.output_dir) / "evidence_highlighter"
    if (highlighter_path / "config.json").exists():
        highlighter = EvidenceHighlighterPredictor(str(highlighter_path))
    else:
        logger.warning("No custom evidence highlighter found. Using base model.")
        highlighter = EvidenceHighlighterPredictor("cross-encoder/ms-marco-MiniLM-L-6-v2")

    verifier_path = Path(args.output_dir) / "contradiction_verifier"
    if (verifier_path / "config.json").exists():
        verifier = ContradictionModelPredictor(str(verifier_path), device)
    else:
        raise RuntimeError("Custom contradiction verifier not found. Train verifier first.")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    model_name_slug = args.base_model.replace("/", "_").replace("-", "_")
    cache_file = Path(args.output_dir) / f"feature_cache_{model_name_slug}.pkl"

    dataset = CombinedHallucinationDataset(
        halueval_samples=halu,
        ragtruth_samples=ragtruth,
        tokenizer=tokenizer,
        highlighter=highlighter,
        verifier=verifier,
        bi_encoder=bi_encoder,
        max_length=384,
        cache_path=str(cache_file)
    )

    num_samples = len(dataset)
    indices = list(range(num_samples))
    random.seed(args.seed)
    random.shuffle(indices)
    split_idx = int(0.9 * num_samples)
    val_indices = indices[split_idx:]
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(val_ds, batch_size=32) # Increased batch size for faster evaluation

    logger.info(f"Loaded {len(val_ds)} validation samples.")

    # 2. Setup the four comparison model paths
    models_info = {
        "Full HaRAG": {
            "path": Path(args.output_dir) / "hallucination_detector_full",
            "ablation": "full",
        },
        "Evidence-only": {
            "path": Path(args.output_dir) / "hallucination_detector_evidence_only",
            "ablation": "evidence_only",
        },
        "Verification-only": {
            "path": Path(args.output_dir) / "hallucination_detector_verification_only",
            "ablation": "verification_only",
        },
        "Standard RAG": {
            "path": Path(args.output_dir) / "hallucination_detector_hallucination_only",
            "ablation": "hallucination_only",
        },
        "Text-only Detector": {
            "path": Path(args.output_dir) / "hallucination_detector_hallucination_only_text",
            "ablation": "hallucination_only_text",
        },
    }

    # Evaluate predictions
    model_predictions = {}
    y_true = []

    # Get labels first
    for batch in val_loader:
        labels = batch["labels"].numpy().tolist()
        y_true.extend(labels)

    for name, info in models_info.items():
        model_path = info["path"] / "model.pt"
        if not model_path.exists():
            # Fall back to confidence_scorer for Full HaRAG if run_all_ablations wasn't used
            if name == "Full HaRAG":
                alt_path = Path(args.output_dir) / "confidence_scorer" / "model.pt"
                if alt_path.exists():
                    model_path = alt_path
            
            if not model_path.exists():
                logger.warning(f"Checkpoint for model {name} not found at {model_path}. Skipping.")
                continue

        logger.info(f"Running inference for model: {name}...")
        model = CombinedHallucinationModel(base_model=args.base_model, ablation_type=info["ablation"])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        preds_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                retrieval_scores = batch["retrieval_score"].to(device)
                evidence_scores = batch["evidence_score"].to(device)
                nli_scores = batch["nli_score"].to(device)

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    retrieval_scores=retrieval_scores,
                    evidence_scores=evidence_scores,
                    nli_scores=nli_scores
                )
                logits = out["logits"].cpu().numpy()
                preds = (logits >= 0.0).astype(int).tolist()
                preds_list.extend(preds)
        
        model_predictions[name] = preds_list

    if "Full HaRAG" not in model_predictions:
        logger.error("Full HaRAG model predictions are required for comparison but checkpoint was not found.")
        return

    # 3. Perform Statistical Tests
    results = {}
    full_preds = model_predictions["Full HaRAG"]
    full_correct = (np.array(full_preds) == np.array(y_true)).astype(int)
    full_acc = accuracy_score(y_true, full_preds)
    full_f1 = f1_score(y_true, full_preds, zero_division=0)

    logger.info("==================================================================================")
    logger.info("STATISTICAL SIGNIFICANCE COMPARISON (Reference: Full HaRAG)")
    logger.info(f"Full HaRAG Accuracy: {full_acc:.4f} | F1: {full_f1:.4f}")
    logger.info("==================================================================================")

    for name, preds in model_predictions.items():
        if name == "Full HaRAG":
            continue

        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        correct = (np.array(preds) == np.array(y_true)).astype(int)

        # Paired t-test (Accuracy)
        t_stat, t_pval = stats.ttest_rel(full_correct, correct)

        # Wilcoxon signed-rank test (Accuracy)
        if np.array_equal(full_correct, correct):
            w_stat, w_pval = 0.0, 1.0
        else:
            try:
                w_stat, w_pval = stats.wilcoxon(full_correct, correct)
            except Exception:
                w_stat, w_pval = 0.0, 1.0

        # McNemar's Test (Binary outcomes)
        mc_stat, mc_pval = mcnemar_test(full_preds, preds, y_true)

        # Bootstrap Confidence Intervals (Accuracy Diff & F1 Diff)
        ci_acc_lower, ci_acc_upper = bootstrap_accuracy_difference(y_true, full_preds, preds)
        ci_f1_lower, ci_f1_upper = bootstrap_f1_difference(y_true, full_preds, preds)

        results[name] = {
            "comparison_model_accuracy": round(float(acc), 4),
            "comparison_model_f1": round(float(f1), 4),
            "accuracy_difference": round(float(full_acc - acc), 4),
            "f1_difference": round(float(full_f1 - f1), 4),
            "paired_t_test": {
                "t_statistic": round(float(t_stat), 4) if not np.isnan(t_stat) else 0.0,
                "p_value": float(t_pval) if not np.isnan(t_pval) else 1.0,
            },
            "wilcoxon_signed_rank": {
                "statistic": round(float(w_stat), 4),
                "p_value": float(w_pval),
            },
            "mcnemar_test": {
                "statistic": round(float(mc_stat), 4),
                "p_value": float(mc_pval),
            },
            "bootstrap_95_accuracy_ci": [round(ci_acc_lower, 4), round(ci_acc_upper, 4)],
            "bootstrap_95_f1_ci": [round(ci_f1_lower, 4), round(ci_f1_upper, 4)]
        }

        logger.info(f"Model: {name}")
        logger.info(f"  Accuracy: {acc:.4f} (Diff vs Full: {full_acc - acc:+.4f}) | F1: {f1:.4f} (Diff: {full_f1 - f1:+.4f})")
        logger.info(f"  McNemar's test        : p-value = {mc_pval:.4f}")
        logger.info(f"  Wilcoxon signed-rank  : p-value = {w_pval:.4f}")
        logger.info(f"  Paired t-test         : p-value = {t_pval:.4f}")
        logger.info(f"  Bootstrap 95% Acc CI  : [{ci_acc_lower:.4f}, {ci_acc_upper:.4f}]")
        logger.info(f"  Bootstrap 95% F1 CI   : [{ci_f1_lower:.4f}, {ci_f1_upper:.4f}]")
        logger.info("----------------------------------------------------------------------------------")

    # Export results
    export_path = Path("results/significance_results.json")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Statistical significance results saved to {export_path}")

if __name__ == "__main__":
    main()
