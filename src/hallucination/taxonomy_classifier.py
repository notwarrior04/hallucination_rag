class TaxonomyClassifier:

    def classify(
        self,
        verification_result,
        mismatches
    ):

        if verification_result["status"] in (
            "SUPPORTED",
            "PARTIALLY_SUPPORTED",
        ):
            return {
                "hallucination": False,
                "type": "NONE"
            }

        labels = []

        for mismatch in mismatches:

            mismatch_type = (
                mismatch["type"]
            )

            mapping = {

                "DATE_MISMATCH":
                    "DATE_HALLUCINATION",

                "NUMERICAL_MISMATCH":
                    "NUMERICAL_HALLUCINATION",

                "ENTITY_MISMATCH":
                    "ENTITY_HALLUCINATION",

                "LOCATION_MISMATCH":
                    "LOCATION_HALLUCINATION",

                "RELATION_MISMATCH":
                    "RELATION_HALLUCINATION"
            }

            labels.append(
                mapping.get(
                    mismatch_type,
                    "UNSUPPORTED_CLAIM"
                )
            )

        if not labels:

            labels.append(
                "UNSUPPORTED_CLAIM"
            )

        return {
            "hallucination": True,
            "type": labels
        }