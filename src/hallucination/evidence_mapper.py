from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EvidenceMapper:
    """
    Claim → Evidence Mapper

    Finds top-k evidence sentences
    most relevant to a claim.
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):

        self.model = SentenceTransformer(
            model_name
        )

    def map_evidence(
        self,
        claim: str,
        evidence_sentences: list,
        top_k: int = 3
    ):

        claim_embedding = self.model.encode(
            [claim]
        )

        evidence_embeddings = self.model.encode(
            evidence_sentences
        )

        scores = cosine_similarity(
            claim_embedding,
            evidence_embeddings
        )[0]

        ranked = sorted(
            zip(
                evidence_sentences,
                scores
            ),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for sentence, score in ranked[:top_k]:

            results.append(
                {
                    "evidence": sentence,
                    "similarity": round(
                        float(score),
                        4
                    )
                }
            )

        return {
            "claim": claim,
            "top_evidence": results
        }
    def get_best_evidence(
        self,
        claim,
        evidence_sentences
    ):
        result = self.map_evidence(
            claim,
            evidence_sentences,
            top_k=1
        )

        return result["top_evidence"][0]