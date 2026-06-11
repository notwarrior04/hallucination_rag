from rag_pipeline import ContradictionVerifier

verifier = ContradictionVerifier(
    "./checkpoints/contradiction_verifier"
)

tests = [
    (
        "Albert Einstein developed the theory of relativity.",
        [{"text": "Albert Einstein developed the theory of relativity."}]
    ),
    (
        "Isaac Newton developed the theory of relativity.",
        [{"text": "Albert Einstein developed the theory of relativity."}]
    ),
    (
        "Relativity is important in physics.",
        [{"text": "Albert Einstein developed the theory of relativity."}]
    )
]

for answer, evidence in tests:
    contradiction, label = verifier.verify(answer, evidence)

    print("=" * 50)
    print("ANSWER:", answer)
    print("CONTRADICTION:", contradiction)
    print("LABEL:", label)