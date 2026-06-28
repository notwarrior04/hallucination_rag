from nltk.tokenize import sent_tokenize

#Extract claims from the answer using sentence tokenization
def extract_claims(answer):
    return sent_tokenize(answer)


#Verify each claim against the evidence using the NLI model
def verify_claim(claim, evidence,nli):

    result = nli(
        {
            "text": evidence,
            "text_pair": claim
        }
    )

    return result


#Calculate the coverage of the answer based on the claims and evidence
def calculate_coverage(answer, evidence, nli):

    claims = extract_claims(answer)

    total_claims = len(claims)

    supported_claims = 0

    results = []

    for claim in claims:

        verification = verify_claim(
            claim,
            evidence,
            nli
        )

        label = verification["label"]

        if label.lower() == "entailment":
            supported_claims += 1

        results.append(
            {
                "claim": claim,
                "label": label,
                "confidence": verification["score"]
            }
        )

    coverage = supported_claims / total_claims

    return {
        "coverage": coverage,
        "supported": supported_claims,
        "total": total_claims,
        "details": results
    }

#Generate the coverage report for the sample answer and evidence
def print_report(report):

    print("="*60)
    print("EVIDENCE COVERAGE REPORT")
    print("="*60)

    for item in report["details"]:

        print("\nClaim:")
        print(item["claim"])

        print("Label:", item["label"])

        print(
            "Confidence:",
            round(item["confidence"],4)
        )

    print("\n" + "="*60)

    print(
        f"Coverage: "
        f"{round(report['coverage']*100,2)}%"
    )

    print(
        f"Supported Claims: "
        f"{report['supported']}"
    )

    print(
        f"Total Claims: "
        f"{report['total']}"
    )
