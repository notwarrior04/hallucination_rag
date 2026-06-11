from rag_pipeline import ContradictionVerifier

v = ContradictionVerifier("./checkpoints/contradiction_verifier")

tests = [
    (
        "Albert Einstein developed the theory of relativity.",
        "Albert Einstein developed the theory of relativity."
    ),
    (
        "Albert Einstein developed the theory of relativity.",
        "Isaac Newton developed the theory of relativity."
    ),
    (
        "Albert Einstein developed the theory of relativity.",
        "Marie Curie developed the theory of relativity."
    ),
    (
        "The capital of France is Paris.",
        "The capital of France is London."
    ),

    # NEW TESTS
    (
        "The capital of France is Paris.",
        "Paris."
    ),
    (
        "The capital of France is Paris.",
        "Paris"
    ),
    (
        "Paris is the capital of France.",
        "Paris"
    ),
]

for p, h in tests:
    print("\n" + "="*50)
    print("PREMISE:", p)
    print("HYPOTHESIS:", h)
    print(v._nli_predict(p, h))