from __future__ import annotations

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class NLIVerifier:
    """
    Natural Language Inference verifier.

    Uses facebook/bart-large-mnli directly instead of the
    transformers pipeline for stable and reproducible inference.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        self.model.eval()

        self.id2label = self.model.config.id2label

    def _predict(
        self,
        premise: str,
        hypothesis: str,
    ):

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():

            logits = self.model(**inputs).logits

            probs = torch.softmax(
                logits,
                dim=1,
            )

        score, idx = torch.max(
            probs,
            dim=1,
        )

        label = self.id2label[
            idx.item()
        ]

        return label, round(
            score.item(),
            4,
        )

    def verify_claim(
        self,
        claim: str,
        evidence_list: list,
    ):

        results = []

        for item in evidence_list:

            evidence = item["evidence"]

            label, score = self._predict(
                premise=evidence,
                hypothesis=claim,
            )

            results.append(
                {
                    "evidence": evidence,
                    "label": label,
                    "score": score,
                }
            )

        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        return results