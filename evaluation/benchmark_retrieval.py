import os
import sys
import json
import random
import logging
import argparse
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from scipy.stats import ttest_rel

# Reconfigure stdout to use UTF-8 to handle unicode characters on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Insert root folder to sys.path so we can import dependencies
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_loader import DatasetLoader
from rag_pipeline import HybridRetriever

def evaluate_retriever(retriever: HybridRetriever, eval_suite: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Evaluates the retriever on the eval_suite.
    Returns:
      - summary_metrics: dict of aggregated metrics
      - per_query_metrics: dict of lists of per-query metrics for significance testing
    """
    metrics = {
        "recall_1": [],
        "recall_5": [],
        "recall_10": [],
        "recall_20": [],
        "mrr": [],
        "ndcg_10": []
    }
    
    for idx, item in enumerate(eval_suite):
        query = item["question"]
        gold_doc_id = item["gold_doc_id"]
        
        # Retrieve top 200 documents
        results, _ = retriever.retrieve(query, top_k=200)
        
        # Deduplicate results by base_doc_id
        seen = set()
        unique_results = []
        for doc in results:
            if doc.base_doc_id not in seen:
                seen.add(doc.base_doc_id)
                unique_results.append(doc)
        
        # Find gold doc rank based on base document ID (deduplicated)
        target_rank = None
        for rank_idx, doc in enumerate(unique_results):
            if doc.base_doc_id == gold_doc_id:
                target_rank = rank_idx + 1
                break
                
        r1 = 1.0 if (target_rank is not None and target_rank == 1) else 0.0
        r5 = 1.0 if (target_rank is not None and target_rank <= 5) else 0.0
        r10 = 1.0 if (target_rank is not None and target_rank <= 10) else 0.0
        r20 = 1.0 if (target_rank is not None and target_rank <= 20) else 0.0
        
        ndcg_10 = 1.0 / np.log2(target_rank + 1) if (target_rank is not None and target_rank <= 10) else 0.0
        mrr = 1.0 / target_rank if target_rank is not None else 0.0
        
        metrics["recall_1"].append(r1)
        metrics["recall_5"].append(r5)
        metrics["recall_10"].append(r10)
        metrics["recall_20"].append(r20)
        metrics["mrr"].append(mrr)
        metrics["ndcg_10"].append(ndcg_10)
        
        if (idx + 1) % 50 == 0 or (idx + 1) == len(eval_suite):
            logger.info(f"Evaluated {idx + 1}/{len(eval_suite)} queries...")
            
    summary_metrics = {
        "Recall@1":   round(float(np.mean(metrics["recall_1"])), 4),
        "Recall@5":   round(float(np.mean(metrics["recall_5"])), 4),
        "Recall@10":  round(float(np.mean(metrics["recall_10"])), 4),
        "Recall@20":  round(float(np.mean(metrics["recall_20"])), 4),
        "MRR":        round(float(np.mean(metrics["mrr"])), 4),
        "nDCG@10":    round(float(np.mean(metrics["ndcg_10"])), 4)
    }
    
    return summary_metrics, metrics

def run_significance_test(baseline_vals: List[float], upgraded_vals: List[float]) -> float:
    """Computes paired t-test p-value. If outputs are constant, returns 1.0."""
    if np.array_equal(baseline_vals, upgraded_vals):
        return 1.0
    t_stat, p_val = ttest_rel(baseline_vals, upgraded_vals)
    return float(p_val) if not np.isnan(p_val) else 1.0

def main():
    parser = argparse.ArgumentParser(description="Run Retrieval Benchmarks separately on SQuAD, FEVER, RAGTruth")
    parser.add_argument("--max_docs", type=int, default=2000, help="Max docs to load per corpus")
    parser.add_argument("--max_queries", type=int, default=200, help="Max queries to evaluate per dataset")
    parser.add_argument("--output_file", type=str, default="results/retrieval_benchmark_results.json", help="Path to save results JSON")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    
    loader = DatasetLoader()
    results = {}

    # Create directories for output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # List of datasets to evaluate
    datasets = ["squad", "fever", "ragtruth"]

    for dataset in datasets:
        logger.info("=" * 80)
        logger.info(f"EVALUATING RETRIEVAL ON DATASET: {dataset.upper()}")
        logger.info("=" * 80)

        corpus = []
        eval_suite = []

        if dataset == "squad":
            # 1. Load SQuAD corpus & queries
            logger.info(f"Loading SQuAD validation corpus (max_docs={args.max_docs})...")
            squad_docs = loader.load_squad_corpus(split="validation", max_docs=args.max_docs)
            corpus = squad_docs
            
            logger.info("Loading SQuAD QA pairs...")
            qa_pairs = loader.load_squad_qa_pairs(split="validation", max_pairs=10000)
            indexed_doc_ids = {doc["doc_id"] for doc in corpus}
            
            # Filter answerable QA pairs where target gold_doc_id is indexed
            eligible_qa = [
                qa for qa in qa_pairs 
                if qa["answerable"] and qa.get("gold_doc_id") in indexed_doc_ids
            ]
            random.shuffle(eligible_qa)
            eval_suite = eligible_qa[:args.max_queries]
            
        elif dataset == "fever":
            # 2. Load FEVER corpus & queries
            logger.info(f"Loading FEVER validation corpus (max_docs={args.max_docs})...")
            fever_docs = loader.load_fever_corpus(split="labelled_dev", max_docs=args.max_docs)
            corpus = fever_docs
            
            # For each indexed document in FEVER corpus, its query is 'claim' and gold is 'doc_id'
            eligible_fever = []
            for doc in corpus:
                if doc.get("claim") and doc.get("doc_id"):
                    eligible_fever.append({
                        "question": doc["claim"],
                        "gold_doc_id": doc["doc_id"]
                    })
            random.shuffle(eligible_fever)
            eval_suite = eligible_fever[:args.max_queries]
            
        elif dataset == "ragtruth":
            # 3. Load RAGTruth corpus & queries
            logger.info(f"Loading RAGTruth test corpus (max_docs={args.max_docs})...")
            raw_ragtruth = loader.load_ragtruth(split="test", max_samples=args.max_docs * 2)
            
            # RAGTruth corpus documents extracted from unique source_info texts
            seen_texts = {}
            ragtruth_docs = []
            eligible_ragtruth = []
            for sample in raw_ragtruth:
                text = sample["text"]
                if not text or not sample.get("prompt"):
                    continue
                if text not in seen_texts:
                    doc_id = f"ragtruth_doc_{len(seen_texts)}"
                    seen_texts[text] = doc_id
                    ragtruth_docs.append({
                        "doc_id": doc_id,
                        "text": text,
                        "source": "ragtruth"
                    })
                
                gold_doc_id = seen_texts[text]
                eligible_ragtruth.append({
                    "question": sample["prompt"],
                    "gold_doc_id": gold_doc_id
                })
            
            corpus = ragtruth_docs[:args.max_docs]
            # Re-filter queries to ensure their gold documents are in our truncated corpus
            final_indexed_ids = {doc["doc_id"] for doc in corpus}
            eligible_ragtruth = [
                q for q in eligible_ragtruth 
                if q["gold_doc_id"] in final_indexed_ids
            ]
            random.shuffle(eligible_ragtruth)
            eval_suite = eligible_ragtruth[:args.max_queries]

        if not corpus or not eval_suite:
            logger.warning(f"No corpus/queries found for dataset {dataset}. Skipping.")
            continue

        logger.info(f"Indexed corpus size: {len(corpus)}")
        logger.info(f"Evaluation queries:  {len(eval_suite)}")

        # A. Baseline Retrieval (No Chunking + Weighted Fusion)
        logger.info("Running Baseline Retrieval...")
        baseline_retriever = HybridRetriever(chunking_config=None, fusion_method="weighted")
        baseline_retriever.index(corpus)
        baseline_summary, baseline_per_q = evaluate_retriever(baseline_retriever, eval_suite)

        # B. Upgraded Retrieval (Chunking + RRF)
        logger.info("Running Upgraded Retrieval...")
        upgraded_retriever = HybridRetriever(
            chunking_config={"chunk_size": 128, "overlap": 32},
            fusion_method="rrf"
        )
        upgraded_retriever.index(corpus)
        upgraded_summary, upgraded_per_q = evaluate_retriever(upgraded_retriever, eval_suite)

        # C. Significance Tests
        significance = {}
        for metric_key in ["recall_5", "recall_10", "mrr"]:
            p_val = run_significance_test(baseline_per_q[metric_key], upgraded_per_q[metric_key])
            significance[f"{metric_key}_p_value"] = round(p_val, 6)

        results[dataset] = {
            "baseline": baseline_summary,
            "upgraded": upgraded_summary,
            "significance": significance
        }

        # Print comparative table for the current dataset
        print("\n" + "="*80)
        print(f"RETRIEVAL COMPARISON: {dataset.upper()}")
        print("="*80)
        print(f"{'Metric':15s} | {'Baseline':10s} | {'Upgraded':10s} | {'Delta':10s}")
        print("-" * 80)
        for m_name in baseline_summary.keys():
            b_val = baseline_summary[m_name]
            u_val = upgraded_summary[m_name]
            delta = u_val - b_val
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            print(f"{m_name:15s} | {b_val:10.4f} | {u_val:10.4f} | {delta_str:10s}")
        print("-" * 80)
        print(f"Paired Significance (p-values):")
        print(f"  Recall@5 p-value:  {significance['recall_5_p_value']:.6f}")
        print(f"  Recall@10 p-value: {significance['recall_10_p_value']:.6f}")
        print(f"  MRR p-value:       {significance['mrr_p_value']:.6f}")
        print("="*80 + "\n")

    # Save all results to the JSON file
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Successfully saved separate retrieval metrics & significance data to {args.output_file}")

if __name__ == "__main__":
    main()
