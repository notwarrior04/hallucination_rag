import json
import argparse
import logging
from pathlib import Path
from rag_pipeline import HallucinationAwareRAG
from data.dataset_loader import DatasetLoader
from sklearn.metrics import roc_auc_score, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["TruthfulQA", "HoVer", "HaluBench"])
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/ood")
    args = parser.parse_args()

    # 1. Pipeline & Data
    rag = HallucinationAwareRAG()
    loader = DatasetLoader()
    
    if args.dataset == "TruthfulQA":
        data = loader.load_truthfulqa()[:args.max_samples]
    elif args.dataset == "HoVer":
        data = loader.load_hover(split="test")[:args.max_samples]
    elif args.dataset == "HaluBench":
        data = loader.load_halubench()[:args.max_samples]
    
    # Simple corpus for retrieval
    corpus = loader.load_squad_corpus(max_docs=500)
    rag.retriever.index(corpus)

    # 2. Evaluation
    logger.info(f"OOD Evaluation on {args.dataset}")
    results = rag.evaluate_dataset(data, dataset_name=args.dataset)

    # 3. Label Mapping and Metrics
    y_true = []
    for d in data:
        if args.dataset == "HoVer":
            # HoVer: 1 is REFUTES, 0 is SUPPORTS (normalized by loader)
            y_true.append(int(d.get("label", 0)))
        elif args.dataset == "HaluBench":
            # HaluBench: 1 is hallucinated, 0 is correct (normalized by loader)
            y_true.append(int(d.get("label", 0)))
        elif args.dataset == "TruthfulQA":
            # TruthfulQA has no binary hallucination label per sample (it's open ended)
            pass
        else:
            y_true.append(int(d.get("label", 0)))
            
    y_prob = [r.hallucination_probability for r in results]
    y_pred = [1 if p > 0.5 else 0 for p in y_prob]
    
    metrics = {
        "dataset": args.dataset,
        "n_samples": len(results)
    }
    
    if y_true and args.dataset != "TruthfulQA":
        if len(set(y_true)) > 1:
            metrics["auc"] = round(roc_auc_score(y_true, y_prob), 4)
        else:
            metrics["auc"] = 0.5
        metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
    else:
        logger.info(f"Skipping factuality metrics for {args.dataset}")

    # 4. Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / f"{args.dataset}_ood_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"OOD Results for {args.dataset}: {metrics}")

if __name__ == "__main__":
    main()
