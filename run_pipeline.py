"""
run_pipeline.py
===============
Simple inference runner for the Hallucination-Aware RAG (HaRAG) pipeline.
"""

import argparse
import logging
import json
from pathlib import Path
from rag_pipeline import HallucinationAwareRAG, Generator, RAGResult
from data.dataset_loader import DatasetLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="HaRAG Inference Runner")
    parser.add_argument("--query", type=str, required=True, help="Question to ask the RAG system")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "evidence_only", "verification_only", "hallucination_only", "hallucination_only_text", "baseline_standard", "baseline_evidence", "baseline_halluc"], help="Pipeline mode")
    parser.add_argument("--output_dir", type=str, default="results/single_run", help="Directory to save output")
    
    # Model/Checkpoint Args
    parser.add_argument("--generator_model", type=str, default="google/flan-t5-base", help="Base generator model")
    parser.add_argument("--highlighter_model", type=str, default="./checkpoints/evidence_highlighter", help="Path to trained highlighter")
    parser.add_argument("--verifier_model", type=str, default="./checkpoints/contradiction_verifier", help="Path to trained verifier")
    parser.add_argument("--hallucination_model", type=str, default="./checkpoints/confidence_scorer", help="Path to trained hallucination detector")
    
    # Corpus Selection
    parser.add_argument("--corpus", type=str, default="squad", choices=["squad", "fever", "combined"], help="Corpus to index")
    parser.add_argument("--max_docs", type=int, default=100, help="Max documents to index for demo")
    parser.add_argument("--pre_generated_answer", type=str, default=None, help="Force a pre-generated answer for verification testing")
    
    args = parser.parse_args()

    # 1. Pipeline Initialization
    logger.info("Initializing HaRAG Pipeline...")
    rag = HallucinationAwareRAG(
        generator=Generator(args.generator_model),
        highlighter_path=args.highlighter_model,
        verifier_path=args.verifier_model,
        hallucination_path=args.hallucination_model
    )
    
    # 2. Dynamic Corpus Selection
    loader = DatasetLoader()
    logger.info(f"Indexing {args.corpus} corpus (max_docs={args.max_docs})...")
    if args.corpus == "squad":
        corpus = loader.load_squad_corpus(split="validation", max_docs=args.max_docs)
    elif args.corpus == "fever":
        corpus = loader.load_fever_corpus(split="labelled_dev", max_docs=args.max_docs)
    else:
        corpus = loader.load_squad_corpus(max_docs=args.max_docs//2) + loader.load_fever_corpus(max_docs=args.max_docs//2)
    
    rag.retriever.index(corpus)

    # 3. Run Inference
    logger.info(f"Running inference for: {args.query}")
    result = rag.run(args.query, mode=args.mode, pre_generated_answer=args.pre_generated_answer)

    # 4. Save and Print Results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print("\n" + "="*60)
    print(f"QUERY: {result.query}")
    print(f"ANSWER: {result.answer}")
    print(f"VCS SCORE: {result.calibrated_vcs:.4f} ({result.hallucination_risk} RISK)")
    print(f"HALLUCINATION: {result.hallucination_label} (Prob: {result.hallucination_probability:.4f})")
    print(f"VERIFICATION: {result.verification_label} (Contradiction: {result.contradiction_score:.4f})")
    print(f"REASON: {result.confidence_explanation.get('reason', 'N/A')}")
    print("="*60)
    logger.info(f"Full trace saved to {output_path / 'result.json'}")

if __name__ == "__main__":
    main()
