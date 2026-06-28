from __future__ import annotations

from typing import List

from src.document.document import Chunk

from src.verification.claim_extractor import ClaimExtractor
from src.verification.nli_verifier import NLIVerifier
from src.verification.claim_verifier import ClaimVerifier

from src.hallucination.evidence_mapper import EvidenceMapper
from src.hallucination.span_localizer import SpanLocalizer
from src.hallucination.taxonomy_classifier import TaxonomyClassifier

from src.scoring.trust_score import TrustScore


class VerificationPipeline:
    """
    Orchestrates the complete verification workflow.

    Answer
        ↓
    Claim Extraction
        ↓
    Evidence Mapping
        ↓
    NLI Verification
        ↓
    Span Localization
        ↓
    Claim Verification
        ↓
    Taxonomy Classification
        ↓
    Trust Score
    """

    def __init__(self):

        self.claim_extractor = ClaimExtractor()

        self.evidence_mapper = EvidenceMapper()

        self.nli = NLIVerifier()

        self.span_localizer = SpanLocalizer()

        self.claim_verifier = ClaimVerifier()

        self.taxonomy = TaxonomyClassifier()

        self.trust = TrustScore()

    def verify(
        self,
        answer_text: str,
        retrieved_chunks: List[Chunk],
    ):

        claims = self.claim_extractor.extract_claims(
            answer_text
        )

        evidence_sentences = []

        for chunk in retrieved_chunks:

            evidence_sentences.append(
                chunk.text.strip()
            )

        verification_results = []

        for item in claims:

            claim = item["claim"]

            best = self.evidence_mapper.get_best_evidence(
                claim,
                evidence_sentences,
            )

            localized = self.span_localizer.localize(
                claim,
                best["evidence"],
            )

            # ----------------------------------------
            # NLI Verification
            # ----------------------------------------

            nli_results = self.nli.verify_claim(
                claim,
                [best],
            )

            best_nli = nli_results[0]

            label = best_nli["label"].upper()

            if label == "ENTAILMENT":

                similarity = best_nli["score"]

            elif label == "CONTRADICTION":

                similarity = 0.0

            else:

                similarity = best["similarity"]

            verification = self.claim_verifier.verify(
                claim=claim,
                evidence=best["evidence"],
                nli_label=best_nli["label"],
                confidence=best_nli["score"],
                mismatches=localized["mismatches"],
            )

            hallucination = self.taxonomy.classify(
                verification,
                localized["mismatches"],
            )

            verification_results.append(
                {
                    "claim": claim,
                    "evidence": best["evidence"],
                    "similarity": similarity,

                    # NLI
                    "nli_label": best_nli["label"],
                    "nli_score": best_nli["score"],

                    # Verification
                    "verification": verification,

                    # Hallucination
                    "hallucination": hallucination,

                    # Localization
                    "entities": localized,
                }
            )

        trust = self.trust.compute(
            [
                item["verification"]
                for item in verification_results
            ]
        )

        return {
            "claims": claims,
            "verification_results": verification_results,
            "trust_score": trust,
        }