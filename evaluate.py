import os
import json
import argparse
import logging
import re
import string
import numpy as np
import pandas as pd
from typing import List, Dict
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from rag_pipeline import HallucinationAwareRAG, RAGResult
from data.dataset_loader import DatasetLoader
from temperature_scaling import compute_ece, compute_brier_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def normalize_answer(s: str) -> str:
    """Lower case, remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_nlp_metrics(predictions: List[str], references: List[str]) -> Dict:
    """SQuAD style F1 and EM."""
    ems = []
    f1s = []
    for pred, ref in zip(predictions, references):
        pred_norm = normalize_answer(pred)
        ref_norm = normalize_answer(ref)
        
        # EM
        ems.append(float(pred_norm == ref_norm))
        
        # F1
        pred_tokens = pred_norm.split()
        ref_tokens = ref_norm.split()
        common = set(pred_tokens) & set(ref_tokens)
        
        if not pred_tokens or not ref_tokens:
            f1s.append(float(pred_tokens == ref_tokens))
            continue
            
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * (precision * recall) / (precision + recall))
            
    return {
        "em": round(np.mean(ems), 4),
        "f1": round(np.mean(f1s), 4)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="RAGTruth", choices=["RAGTruth", "HaluEval", "SQuAD_v2"], help="Evaluation dataset")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    parser.add_argument("--mode", type=str, default="full")
    args = parser.parse_args()

    # 1. Load Data
    loader = DatasetLoader()
    if args.dataset == "RAGTruth":
        data = loader.load_ragtruth(split=args.split, max_samples=args.max_samples)
        corpus = loader.load_squad_corpus(max_docs=1000)
    elif args.dataset == "HaluEval":
        # Load all HaluEval subsets for the paper
        qa = loader.load_halueval("qa_samples")
        dial = loader.load_halueval("dialogue_samples")
        summ = loader.load_halueval("summarization_samples")
        data = (qa + dial + summ)[:args.max_samples]
        corpus = loader.load_squad_corpus(max_docs=1000)
    else:
        data = []
        corpus = []

    # 2. Pipeline Init
    rag = HallucinationAwareRAG()
    rag.retriever.index(corpus)

    # 3. Batch Inference
    logger.info(f"Evaluating {len(data)} samples from {args.dataset}")
    results = rag.evaluate_dataset(data, mode=args.mode, dataset_name=args.dataset)

    # 4. Metric Computation
    final_metrics = {"mode": args.mode, "dataset": args.dataset}

    # Factuality/Hallucination Labels
    y_true = []
    for d in data:
        if args.dataset == "RAGTruth":
            y_true.append(1 if len(d.get("labels", [])) > 0 else 0)
        else:
            y_true.append(int(d.get("label", 0)))
            
    y_prob = [r.hallucination_probability for r in results]
    y_pred = [1 if p > 0.5 else 0 for p in y_prob]
    
    if y_true:
        final_metrics["accuracy"] = accuracy_score(y_true, y_pred)
        final_metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        final_metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        final_metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        try:
            final_metrics["auc"] = roc_auc_score(y_true, y_prob)
        except:
            final_metrics["auc"] = 0.5
        
        factuality_labels = 1 - np.array(y_true)
        correctness_probs = 1.0 - np.array(y_prob)
        final_metrics["ece"] = compute_ece(correctness_probs, factuality_labels)
        final_metrics["brier"] = compute_brier_score(correctness_probs, factuality_labels)

    # Retrieval Metrics
    final_metrics["mrr"] = np.mean([r.mrr for r in results if r.mrr is not None]) if any(r.mrr is not None for r in results) else 0.0
    final_metrics["hit_rate"] = np.mean([r.hit_rate for r in results if r.hit_rate is not None]) if any(r.hit_rate is not None for r in results) else 0.0

    # NLP Metrics
    if args.dataset != "RAGTruth":
        preds = [r.answer for r in results]
        refs = [d.get("answer", d.get("right", d.get("response", ""))) for d in data]
        nlp = compute_nlp_metrics(preds, refs)
        final_metrics["nlp_em"] = nlp["em"]
        final_metrics["nlp_f1"] = nlp["f1"]
    else:
        logger.info("Skipping NLP metrics for RAGTruth (no gold references)")

    # 5. Save Results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / f"{args.dataset}_{args.mode}_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save ROC Data for plots.py
    from rag_pipeline import save_roc_data
    if y_true:
        save_roc_data(results, [1 - t for t in y_true], out_dir / f"{args.dataset}_roc_data.csv")

    logger.info(f"Evaluation complete for {args.dataset}. Metrics: {final_metrics}")

if __name__ == "__main__":
    main()
