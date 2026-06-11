import os
import sys
import json
import random
import logging
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from data.dataset_loader import DatasetLoader
from rag_pipeline import HybridRetriever

# Reconfigure stdout to use UTF-8 to handle unicode characters on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def evaluate_retriever(retriever: HybridRetriever, eval_suite: List[Dict]) -> Tuple[Dict, List[Dict]]:
    recalls_at_1 = []
    recalls_at_5 = []
    recalls_at_10 = []
    recalls_at_20 = []
    
    hits_at_1 = []
    hits_at_5 = []
    hits_at_10 = []
    hits_at_20 = []
    
    ndcgs_at_10 = []
    successes_at_100 = []
    mrrs = []
    ranks = []
    
    detailed_results = []
    
    for idx, item in enumerate(eval_suite):
        query = item["question"]
        gold_doc_id = item["gold_doc_id"]
        
        # Retrieve top 200 documents to compute standard recall ranges
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
                
        recall_1 = 1.0 if (target_rank is not None and target_rank == 1) else 0.0
        recall_5 = 1.0 if (target_rank is not None and target_rank <= 5) else 0.0
        recall_10 = 1.0 if (target_rank is not None and target_rank <= 10) else 0.0
        recall_20 = 1.0 if (target_rank is not None and target_rank <= 20) else 0.0
        
        hit_1 = recall_1
        hit_5 = recall_5
        hit_10 = recall_10
        hit_20 = recall_20
        
        # Single relevant document => IDCG = 1
        ndcg_10 = 1.0 / np.log2(target_rank + 1) if (target_rank is not None and target_rank <= 10) else 0.0
        mrr = 1.0 / target_rank if target_rank is not None else 0.0
        success_100 = 1.0 if (target_rank is not None and target_rank <= 100) else 0.0
        
        recalls_at_1.append(recall_1)
        recalls_at_5.append(recall_5)
        recalls_at_10.append(recall_10)
        recalls_at_20.append(recall_20)
        
        hits_at_1.append(hit_1)
        hits_at_5.append(hit_5)
        hits_at_10.append(hit_10)
        hits_at_20.append(hit_20)
        
        ndcgs_at_10.append(ndcg_10)
        successes_at_100.append(success_100)
        mrrs.append(mrr)
        
        # Mean rank tracks actual index (if not found in top 200, we count it as 201 for safety)
        ranks.append(target_rank if target_rank is not None else 201)
        
        detailed_results.append({
            "query": query,
            "gold_doc_id": gold_doc_id,
            "rank": target_rank if target_rank is not None else -1,
            "recall@5": int(recall_5),
            "mrr": round(mrr, 4),
            "success@100": int(success_100)
        })
        
        if (idx + 1) % 100 == 0 or (idx + 1) == len(eval_suite):
            logger.info(f"Evaluated {idx + 1}/{len(eval_suite)} queries...")
            
    summary_metrics = {
        "Recall@1":   round(float(np.mean(recalls_at_1)), 4),
        "Recall@5":   round(float(np.mean(recalls_at_5)), 4),
        "Recall@10":  round(float(np.mean(recalls_at_10)), 4),
        "Recall@20":  round(float(np.mean(recalls_at_20)), 4),
        "Hit@1":      round(float(np.mean(hits_at_1)), 4),
        "Hit@5":      round(float(np.mean(hits_at_5)), 4),
        "Hit@10":     round(float(np.mean(hits_at_10)), 4),
        "Hit@20":     round(float(np.mean(hits_at_20)), 4),
        "MRR":        round(float(np.mean(mrrs)), 4),
        "nDCG@10":    round(float(np.mean(ndcgs_at_10)), 4),
        "Success@100": round(float(np.mean(successes_at_100)), 4),
        "Mean Rank":  round(float(np.mean(ranks)), 2)
    }
    
    return summary_metrics, detailed_results

def main():
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # 1. Load 5000 combined documents
    loader = DatasetLoader()
    logger.info("Loading 2500 SQuAD v2 documents...")
    squad_docs = loader.load_squad_corpus(split="validation", max_docs=2500)
    
    logger.info("Loading 2500 FEVER documents...")
    fever_docs = loader.load_fever_corpus(split="labelled_dev", max_docs=2500)
    
    # Strictly natural corpus
    corpus = squad_docs + fever_docs
    logger.info(f"Loaded {len(corpus)} combined documents from SQuAD and FEVER.")
    
    # Shuffle the corpus deterministically to mix SQuAD and FEVER
    random.seed(42)
    random.shuffle(corpus)
    
    # 2. Load QA pairs and filter/construct eval suite
    qa_pairs = loader.load_squad_qa_pairs(split="validation", max_pairs=10000)
    
    # Map corpus doc_ids for quick lookup
    indexed_doc_ids = {doc["doc_id"] for doc in corpus}
    
    # Filter answerable QA pairs where their context doc_id is in our corpus
    eligible_qa = []
    for qa in qa_pairs:
        if qa["answerable"] and qa.get("gold_doc_id") in indexed_doc_ids:
            eligible_qa.append(qa)
            
    # Shuffle qa pairs to remove ordering bias deterministically
    random.seed(42)
    random.shuffle(eligible_qa)
    
    logger.info(f"Eligible QA pairs: {len(eligible_qa)}")
    
    # Select MAX_EVAL_QUERIES = 500
    MAX_EVAL_QUERIES = 500
    eval_suite = eligible_qa[:MAX_EVAL_QUERIES]
    
    logger.info(f"Constructed natural evaluation benchmark with {len(eval_suite)} queries.")
    
    # Run Baseline Evaluator
    logger.info("=" * 80)
    logger.info("RUNNING BASELINE EVALUATION (Passage Retrieval + Weighted Fusion)")
    logger.info("=" * 80)
    baseline_retriever = HybridRetriever(chunking_config=None, fusion_method="weighted")
    baseline_retriever.index(corpus)
    baseline_metrics, baseline_details = evaluate_retriever(baseline_retriever, eval_suite)
    
    # Save baseline metrics
    baseline_out = Path("results/baseline_metrics.json")
    baseline_out.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_out, "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    
    # Save baseline details
    baseline_details_out = Path("results/baseline_details.json")
    with open(baseline_details_out, "w") as f:
        json.dump(baseline_details, f, indent=2)
        
    # Run Upgraded Evaluator
    logger.info("=" * 80)
    logger.info("RUNNING UPGRADED EVALUATION (Chunk Retrieval + RRF)")
    logger.info("=" * 80)
    upgraded_retriever = HybridRetriever(
        chunking_config={"chunk_size": 128, "overlap": 32},
        fusion_method="rrf"
    )
    upgraded_retriever.index(corpus)
    upgraded_metrics, upgraded_details = evaluate_retriever(upgraded_retriever, eval_suite)
    
    # Save upgraded details
    upgraded_details_out = Path("results/upgraded_details.json")
    with open(upgraded_details_out, "w") as f:
        json.dump(upgraded_details, f, indent=2)
        
    # Print statistics of the upgraded retriever corpus chunks
    num_docs = len(corpus)
    num_chunks = len(upgraded_retriever.corpus)
    avg_chunks_per_doc = num_chunks / num_docs if num_docs else 0.0
    
    chunk_word_counts = [len(c["text"].split()) for c in upgraded_retriever.corpus]
    avg_chunk_length = np.mean(chunk_word_counts) if chunk_word_counts else 0.0
    min_chunk_length = np.min(chunk_word_counts) if chunk_word_counts else 0
    max_chunk_length = np.max(chunk_word_counts) if chunk_word_counts else 0
    
    print("\n" + "="*80)
    print("CORPUS & BENCHMARK STATISTICS")
    print("="*80)
    print(f"Total Documents:             {num_docs}")
    print(f"Total Chunks (Upgraded):     {num_chunks}")
    print(f"Avg Chunks per Document:     {avg_chunks_per_doc:.2f}")
    print(f"Avg Chunk Length (words):    {avg_chunk_length:.1f}")
    print(f"Min Chunk Length (words):    {min_chunk_length}")
    print(f"Max Chunk Length (words):    {max_chunk_length}")
    print(f"Total Evaluation Queries:    {len(eval_suite)}")
    print("="*80)
    
    # Calculate comparative metrics and delta differences
    nested_metrics = {}
    for metric_name in baseline_metrics.keys():
        b_val = baseline_metrics[metric_name]
        u_val = upgraded_metrics[metric_name]
        delta = u_val - b_val
        nested_metrics[metric_name] = {
            "baseline": b_val,
            "upgraded": u_val,
            "delta": round(delta, 4)
        }
        
    # Save comparative metrics nested JSON
    retrieval_metrics_out = Path("results/retrieval_metrics.json")
    with open(retrieval_metrics_out, "w") as f:
        json.dump(nested_metrics, f, indent=2)
        
    # Print side-by-side comparison table
    print("\n" + "="*80)
    print("COMPARATIVE RETRIEVAL PERFORMANCE METRICS")
    print("="*80)
    print(f"{'Metric':15s} | {'Baseline':10s} | {'Upgraded':10s} | {'Delta':10s}")
    print("-" * 80)
    for m_name, vals in nested_metrics.items():
        delta_str = f"+{vals['delta']:.4f}" if vals['delta'] >= 0 else f"{vals['delta']:.4f}"
        # Mean Rank delta direction is negative (lower is better)
        if m_name == "Mean Rank":
            delta_str = f"{vals['delta']:.2f}" if vals['delta'] <= 0 else f"+{vals['delta']:.2f}"
            print(f"{m_name:15s} | {vals['baseline']:10.2f} | {vals['upgraded']:10.2f} | {delta_str:10s}")
        else:
            print(f"{m_name:15s} | {vals['baseline']:10.4f} | {vals['upgraded']:10.4f} | {delta_str:10s}")
    print("="*80 + "\n")
    logger.info(f"Saved aggregated metrics to {retrieval_metrics_out}")

if __name__ == "__main__":
    main()
