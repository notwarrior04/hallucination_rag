class ClaimVerifier:

    def __init__(
        self,
        similarity_threshold=0.85
    ):

        self.similarity_threshold = (
            similarity_threshold
        )

    def verify(
            self,
            claim,
            evidence,
            nli_label,
            confidence,
            mismatches,
        ):

            if mismatches:
                return {
                    "claim": claim,
                    "status": "CONTRADICTED",
                    "confidence": confidence,
                    "reason": mismatches,
                }

            label = nli_label.upper()

            if label == "ENTAILMENT":
                status = "SUPPORTED"

            elif label == "CONTRADICTION":
                status = "CONTRADICTED"

            else:
                status = "UNSUPPORTED"

            return {
                "claim": claim,
                "status": status,
                "confidence": round(confidence, 4),
                "reason": [],
            }