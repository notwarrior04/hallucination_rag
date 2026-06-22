"""
Dataset Loader
==============
Downloads and preprocesses:
  - SQuAD v2      → QA + unanswerable questions
  - FEVER         → Evidence + Contradiction labels
  - HaluEval      → Hallucination examples (QA, Dialog, Summarization)

Dataset Download Links (also printed at runtime):
  SQuAD v2  : https://rajpurkar.github.io/SQuAD-explorer/
              HuggingFace: datasets.load_dataset("rajpurkar/squad_v2")
  FEVER     : https://fever.ai/dataset/fever.html
              HuggingFace: datasets.load_dataset("copenlu/fever_gold_evidence")
  HaluEval  : https://github.com/RUCAIBox/HaluEval
              HuggingFace: datasets.load_dataset("pminervini/HaluEval", "qa_samples")
  RAGTruth  : https://github.com/ParticleMedia/RAGTruth
              HuggingFace: datasets.load_dataset("wandb/RAGTruth")
  TruthfulQA: https://huggingface.co/datasets/truthful_qa
  HoVer     : https://hover-nlp.github.io/
  HaluBench : https://huggingface.co/datasets/PatronusAI/HaluBench
"""

import logging
import random
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DOWNLOAD_LINKS = {
    "squad_v2": {
        "paper":        "https://arxiv.org/abs/1806.03822",
        "hf":           "https://huggingface.co/datasets/rajpurkar/squad_v2",
        "direct_train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
        "direct_dev":   "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    },
    "fever": {
        "paper":        "https://arxiv.org/abs/1803.05355",
        "hf":           "https://huggingface.co/datasets/copenlu/fever_gold_evidence",
        "direct_train": "https://fever.ai/download/fever/train.jsonl",
        "direct_dev":   "https://fever.ai/download/fever/shared_task_dev.jsonl",
        "wiki_pages":   "https://fever.ai/download/fever/wiki-pages.zip",
    },
    "halueval": {
        "paper":   "https://arxiv.org/abs/2305.11747",
        "github":  "https://github.com/RUCAIBox/HaluEval",
        "hf":      "https://huggingface.co/datasets/pminervini/HaluEval",
        "subsets": ["qa_samples", "dialogue_samples", "summarization_samples"],
    },
    "ragtruth": {
        "github": "https://github.com/ParticleMedia/RAGTruth",
        "hf":     "https://huggingface.co/datasets/wandb/RAGTruth",
        "paper":  "https://arxiv.org/abs/2401.00396",
    },
    "truthfulqa": {
        "hf":    "https://huggingface.co/datasets/truthful_qa",
        "paper": "https://arxiv.org/abs/2109.07958",
    },
    "hover": {
        "hf":      "https://huggingface.co/datasets/hover",
        "website": "https://hover-nlp.github.io/",
        "paper":   "https://arxiv.org/abs/2011.03088",
    },
    "halubench": {
        "hf":    "https://huggingface.co/datasets/PatronusAI/HaluBench",
        "paper": "https://arxiv.org/abs/2408.11338",
    },
}

# FEVER v2.0 (copenlu/fever_gold_evidence) available splits
FEVER_SPLITS = {
    "train":      "train",
    "validation": "validation",
    "test":       "test",
    # legacy aliases → mapped to nearest equivalent
    "labelled_dev":    "validation",
    "shared_task_dev": "validation",
    "shared_task_test": "test",
}


def _resolve_fever_split(split: str) -> str:
    """Resolve legacy FEVER split names to copenlu/fever_gold_evidence split names."""
    resolved = FEVER_SPLITS.get(split, split)
    if resolved != split:
        logger.info(f"Resolved FEVER split '{split}' → '{resolved}'")
    return resolved


class DatasetLoader:
    """
    Unified loader that fetches datasets via HuggingFace `datasets`
    and converts them into a common corpus format:
      [{"doc_id": str, "text": str, "source": str, "label": str, ...}]
    """

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._print_links()

    def _print_links(self):
        logger.info("=" * 60)
        logger.info("DATASET DOWNLOAD LINKS")
        for name, links in DOWNLOAD_LINKS.items():
            logger.info(f"  [{name.upper()}]")
            for k, v in links.items():
                if isinstance(v, str):
                    logger.info(f"    {k:15s}: {v}")
        logger.info("=" * 60)

    # ── SQuAD v2 ──────────────────────────────────────────────

    def load_squad_corpus(
        self,
        split: str = "train",
        max_docs: int = 20000,
    ) -> List[Dict]:
        """Returns passage-level documents from SQuAD v2."""
        from datasets import load_dataset

        logger.info(f"Loading SQuAD v2 [{split}] ...")
        ds = load_dataset(
            "rajpurkar/squad_v2",
            split=split,
            cache_dir=str(self.cache_dir),
        )

        # Shuffle corpus before cutting
        rows = list(ds)
        rng = random.Random(42)
        rng.shuffle(rows)

        seen_contexts = {}
        for row in rows:
            ctx = row["context"]
            if ctx not in seen_contexts:
                import hashlib
                ctx_hash = hashlib.md5(ctx.strip().encode("utf-8")).hexdigest()
                seen_contexts[ctx] = {
                    "doc_id": f"squad_{ctx_hash}",
                    "text":   ctx,
                    "source": "squad_v2",
                    "title":  row.get("title", ""),
                }
            if len(seen_contexts) >= max_docs:
                break

        corpus = list(seen_contexts.values())
        logger.info(f"SQuAD v2 corpus: {len(corpus)} passages")
        return corpus

    def load_squad_qa_pairs(
        self,
        split: str = "train",
        max_pairs: int = 10000,
    ) -> List[Dict]:
        """Returns QA pairs including unanswerable questions."""
        from datasets import load_dataset

        ds = load_dataset(
            "rajpurkar/squad_v2",
            split=split,
            cache_dir=str(self.cache_dir),
        )

        pairs = []
        for row in ds:
            answers = row["answers"]["text"]
            import hashlib
            ctx_hash = hashlib.md5(row["context"].strip().encode("utf-8")).hexdigest()
            pairs.append({
                "id":         row["id"],
                "question":   row["question"],
                "context":    row["context"],
                "gold_doc_id": f"squad_{ctx_hash}",
                "answers":    answers,
                "answerable": len(answers) > 0,
                "source":     "squad_v2",
            })
            if len(pairs) >= max_pairs:
                break
        logger.info(f"SQuAD v2 QA pairs: {len(pairs)}")
        return pairs

    # ── FEVER ─────────────────────────────────────────────────
    # Uses copenlu/fever_gold_evidence — a clean Parquet-based mirror
    # with no loading script. Schema:
    #   id, claim, label, evidence (list of {wikipedia_url, sentence})

    def load_fever_corpus(
        self,
        split: str = "train",
        max_docs: int = 20000,
    ) -> List[Dict]:
        """Returns FEVER claim + evidence as documents."""
        from datasets import load_dataset

        split = _resolve_fever_split(split)
        logger.info(f"Loading FEVER [{split}] via copenlu/fever_gold_evidence ...")
        ds = load_dataset(
            "copenlu/fever_gold_evidence",
            split=split,
            cache_dir=str(self.cache_dir),
        )

        # Shuffle corpus before cutting
        rows = list(ds)
        rng = random.Random(42)
        rng.shuffle(rows)

        docs = []
        for row in rows:
            if len(docs) >= max_docs:
                break

            # Flatten evidence list → single text block
            evidence_parts = []
            for ev in row.get("evidence", []):
                if isinstance(ev, dict):
                    sent = ev.get("sentence", ev.get("evidence_sentence", ""))
                elif isinstance(ev, (list, tuple)):
                    if len(ev) >= 3:
                        sent = str(ev[2])      # actual FEVER evidence sentence
                    elif len(ev) >= 2:
                        sent = str(ev[1])
                    else:
                        sent = str(ev[0])
                else:
                    sent = str(ev)
                if sent:
                    evidence_parts.append(sent)

            ev_text = " ".join(evidence_parts).strip()
            if not ev_text:
                continue

            docs.append({
                "doc_id": f"fever_{len(docs)}",
                "text":   ev_text,
                "label":  row.get("label", "NOT ENOUGH INFO"),
                "claim":  row["claim"],
                "source": "fever",
            })

        logger.info(f"FEVER corpus: {len(docs)} claims")
        return docs

    def load_fever_pairs(
        self,
        split: str = "train",
        max_pairs: int = 5000,
    ) -> List[Dict]:
        """Returns (claim, evidence, label) triplets."""
        from datasets import load_dataset

        split = _resolve_fever_split(split)
        logger.info(f"Loading FEVER pairs [{split}] via copenlu/fever_gold_evidence ...")
        ds = load_dataset(
            "copenlu/fever_gold_evidence",
            split=split,
            cache_dir=str(self.cache_dir),
        )

        pairs = []
        for row in ds:
            evidence_parts = []
            for ev in row.get("evidence", []):
                if isinstance(ev, dict):
                    sent = ev.get("sentence", ev.get("evidence_sentence", ""))
                elif isinstance(ev, (list, tuple)):
                    if len(ev) >= 3:
                        sent = str(ev[2])      # actual FEVER evidence sentence
                    elif len(ev) >= 2:
                        sent = str(ev[1])
                    else:
                        sent = str(ev[0])
                else:
                    sent = str(ev)
                if sent:
                    evidence_parts.append(sent)

            pairs.append({
                "claim":    row["claim"],
                "label":    row.get("label", "NOT ENOUGH INFO"),
                "evidence": evidence_parts,
            })
            if len(pairs) >= max_pairs:
                break

        logger.info(f"FEVER pairs: {len(pairs)}")
        return pairs

    # ── HaluEval ──────────────────────────────────────────────

    def load_halueval(
        self,
        subset: str = "qa_samples",
        split: str = "data",
        max_samples: int = 5000,
    ) -> List[Dict]:
        """
        subsets: qa_samples | dialogue_samples | summarization_samples
        """
        from datasets import load_dataset

        logger.info(f"Loading HaluEval [{subset}] ...")
        ds = load_dataset(
            "pminervini/HaluEval",
            subset,
            split=split,
            cache_dir=str(self.cache_dir),
        )

        samples = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            
            # Context
            text = row.get("knowledge", row.get("dialogue_history", row.get("document", "")))
            
            # Try to get right and hallucinated answers (paired format)
            right = row.get("right_answer", row.get("right_response", row.get("right_summary", "")))
            halluc = row.get("hallucinated_answer", row.get("hallucinated_response", row.get("hallucinated_summary", "")))
            
            # If they are not present, check for unpaired samples format (supports answer, response, summary)
            if not right and not halluc:
                ans = row.get("answer", row.get("response", row.get("summary", "")))
                if ans:
                    is_halluc = row.get("hallucination", "")
                    if str(is_halluc).lower() in ("yes", "fail", "true", "1"):
                        halluc = ans
                        right = ""
                    else:
                        right = ans
                        halluc = ""

            samples.append({
                "id":     f"halueval_{subset}_{i}",
                "text":   text,
                "right":  right,
                "halluc": halluc,
                "source": f"halueval_{subset}",
            })
        logger.info(f"HaluEval [{subset}]: {len(samples)} samples")
        return samples

    # ── RAGTruth ──────────────────────────────────────────────

    def load_ragtruth(
        self,
        split: str = "train",
        max_samples: int = 3000,
    ) -> List[Dict]:
        from datasets import load_dataset

        logger.info(f"Loading RAGTruth [{split}] ...")
        try:
            hf_split = "train" if split == "train" else "test"
            ds = load_dataset(
                "leobianco/ragtruth",
                split=hf_split,
                cache_dir=str(self.cache_dir),
            )
        except Exception as e:
            logger.warning(f"RAGTruth load failed: {e}. Skipping.")
            return []

        samples = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            samples.append({
                "id":       f"ragtruth_{row.get('id', i)}",
                "text":     row.get("source_info", ""),
                "response": row.get("response", ""),
                "labels":   row.get("labels", []),
                "label":    1 if len(row.get("labels", [])) > 0 else 0,
                "prompt":   row.get("prompt", ""),
                "source":   "ragtruth",
            })
        logger.info(f"RAGTruth: {len(samples)} samples")
        return samples

    # ── Out-of-Distribution ───────────────────────────────────

    def load_truthfulqa(self, max_samples: int = 817) -> List[Dict]:
        from datasets import load_dataset
        logger.info("Loading TruthfulQA via rahmanidashti/truthful-qa ...")
        try:
            ds = load_dataset(
                "rahmanidashti/truthful-qa",
                data_files="generation/truthfulqa_gen.parquet",
                cache_dir=str(self.cache_dir),
            )["train"]
        except Exception as e:
            logger.warning(f"TruthfulQA load failed: {e}. Skipping.")
            return []
        return [
            {
                "id":          f"tqa_{i}",
                "question":    row["question"],
                "best_answer": row["best_answer"],
                "source":      "truthfulqa",
            }
            for i, row in enumerate(ds)
            if i < max_samples
        ]

    def load_halubench(self, max_samples: int = 1000) -> List[Dict]:
        from datasets import load_dataset
        logger.info("Loading HaluBench via PatronusAI/HaluBench ...")
        try:
            ds = load_dataset(
                "PatronusAI/HaluBench",
                cache_dir=str(self.cache_dir),
            )["test"]
        except Exception as e:
            logger.warning(f"HaluBench load failed: {e}. Skipping.")
            return []
        return [
            {
                "id":       f"halubench_{i}",
                "text":     row.get("passage", ""),
                "question": row.get("question", ""),
                "answer":   row.get("answer", ""),
                "label":    1 if row.get("label", "") == "FAIL" else 0,
                "source":   row.get("source_ds", ""),
            }
            for i, row in enumerate(ds)
            if i < max_samples
        ]

    def load_hover(self, split: str = "train", max_samples: int = 5000) -> List[Dict]:
        # Lazy load datasets package
        from datasets import load_dataset
        logger.info(f"Loading HoVer [{split}] via Dzeniks/hover ...")
        try:
            ds = load_dataset(
                "Dzeniks/hover",
                split=split,
                cache_dir=str(self.cache_dir),
            )
        except Exception as e:
            logger.warning(f"HoVer load failed: {e}. Skipping.")
            return []

        samples = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            ev = row.get("evidence", "")
            if isinstance(ev, list):
                ev = "\n".join(
                    str(x[-1]) if isinstance(x, (list, tuple)) else str(x)
                    for x in ev
                )
            samples.append({
                "id":       f"hover_{i}",
                "claim":    row.get("claim", ""),
                "evidence": ev,
                "label":    row.get("label", 0),
                "source":   "hover",
            })
        logger.info(f"HoVer: {len(samples)} samples")
        return samples

    def get_dataset_statistics(self) -> Dict:
        """Returns statistics for all available datasets."""
        stats = {
            "squad_v2": {"type": "QA", "metric": "F1/EM"},
            "fever": {"type": "Fact Verification", "metric": "Accuracy"},
            "halueval": {"type": "Hallucination Detection", "metric": "Accuracy"},
            "ragtruth": {"type": "RAG Hallucination", "metric": "Binary/NLI"},
            "truthfulqa": {"type": "OOD Truthfulness", "metric": "MC1/MC2"},
            "hover": {"type": "Multihop Verification", "metric": "Accuracy"},
            "halubench": {"type": "Benchmarking", "metric": "Pass/Fail"}
        }
        return stats


    # ── Combined training corpus ──────────────────────────────

    def build_training_corpus(self) -> List[Dict]:
        """All in-distribution training documents fused into one corpus."""
        corpus = (
            self.load_squad_corpus(split="train", max_docs=10000)
            + self.load_fever_corpus(split="train", max_docs=5000)
        )
        logger.info(f"Total training corpus size: {len(corpus)}")
        return corpus


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DatasetLoader()

    print("\n--- SQuAD v2 ---")
    squad = loader.load_squad_corpus(max_docs=100)
    print(f"Loaded {len(squad)} passages")
    print("Sample:", squad[0])

    print("\n--- FEVER ---")
    fever = loader.load_fever_pairs(split="train", max_pairs=100)
    print(f"Loaded {len(fever)} pairs")
    print("Sample:", fever[0])

    print("\n--- HaluEval ---")
    halu = loader.load_halueval(max_samples=100)
    print(f"Loaded {len(halu)} samples")
    print("Sample:", halu[0])

    print(f"\nSQuAD: {len(squad)}, FEVER: {len(fever)}, HaluEval: {len(halu)}")
    print("\nAll datasets loaded successfully!")