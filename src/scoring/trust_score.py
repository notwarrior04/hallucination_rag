class TrustScore:

    def compute(
        self,
        verification_results,
    ):

        total = len(verification_results)

        if total == 0:

            return {
                "score": 0,
                "supported": 0,
                "partial": 0,
                "unsupported": 0,
                "contradicted": 0,
            }

        score = 0

        supported = 0
        partial = 0
        unsupported = 0
        contradicted = 0

        for item in verification_results:

            status = item["status"]

            if status == "SUPPORTED":

                supported += 1
                score += 1

            elif status == "PARTIALLY_SUPPORTED":

                partial += 1
                score += 0.5

            elif status == "UNSUPPORTED":

                unsupported += 1

            elif status == "CONTRADICTED":

                contradicted += 1
                score -= 1

        score = max(score, 0)

        score = (score / total) * 100

        return {
            "score": round(score, 2),
            "supported": supported,
            "partial": partial,
            "unsupported": unsupported,
            "contradicted": contradicted,
        }