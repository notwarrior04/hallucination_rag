from __future__ import annotations


class ExplanationGenerator:

    def generate(
        self,
        answer,
        verification_report,
    ):

        lines = []

        lines.append("=" * 80)
        lines.append("GENERATED ANSWER")
        lines.append("=" * 80)
        lines.append(answer.text)
        lines.append("")

        lines.append("=" * 80)
        lines.append("CLAIM VERIFICATION")
        lines.append("=" * 80)

        supported = 0
        partial = 0
        unsupported = 0
        contradicted = 0
        hallucinations = 0

        for idx, item in enumerate(
            verification_report["verification_results"],
            start=1,
        ):

            verification = item["verification"]
            hallucination = item["hallucination"]

            status = verification["status"]

            if status == "SUPPORTED":
                supported += 1

            elif status == "PARTIALLY_SUPPORTED":
                partial += 1

            elif status == "UNSUPPORTED":
                unsupported += 1

            elif status == "CONTRADICTED":
                contradicted += 1

            if hallucination["hallucination"]:
                hallucinations += 1

            lines.append(f"\nClaim {idx}")
            lines.append("-" * 80)

            lines.append(item["claim"])
            lines.append("")

            lines.append(f"Evidence        : {item['evidence']}")
            lines.append(f"Similarity      : {item['similarity']:.4f}")

            lines.append(
                f"NLI             : "
                f"{item['nli_label']} "
                f"({item['nli_score']:.4f})"
            )

            lines.append(
                f"Verification    : {status}"
            )

            lines.append(
                f"Hallucination   : "
                f"{hallucination['hallucination']}"
            )

            lines.append(
                f"Type            : "
                f"{hallucination['type']}"
            )

            if verification["reason"]:

                lines.append(
                    f"Reason          : "
                    f"{verification['reason']}"
                )

        lines.append("")
        lines.append("=" * 80)
        lines.append("SUMMARY")
        lines.append("=" * 80)

        lines.append(f"Supported Claims        : {supported}")
        lines.append(f"Partially Supported     : {partial}")
        lines.append(f"Unsupported Claims      : {unsupported}")
        lines.append(f"Contradicted Claims     : {contradicted}")
        lines.append(f"Hallucinations Detected : {hallucinations}")

        lines.append("")

        lines.append("=" * 80)
        lines.append("TRUST SCORE")
        lines.append("=" * 80)

        trust = verification_report["trust_score"]

        if isinstance(trust, dict):
            score = trust["score"]
        else:
            score = trust

        lines.append(f"{score}%")

        return "\n".join(lines)