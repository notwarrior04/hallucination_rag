from pathlib import Path

from src.pipelines.rag_pipeline import RAGPipeline

from src.retrieval.retriever import Retriever

from src.generation.llm_interface import LLMInterface
from src.generation.rag_generator import RAGGenerator

from src.verification.verification_pipeline import VerificationPipeline
from src.explainability.explanation_generator import ExplanationGenerator


# ============================================================
# Configuration
# ============================================================

PDF_DIRECTORY = Path("data/raw")

LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-2.5-flash"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("EHD-RAG")
    print("Explainable Hybrid Document Retrieval-Augmented Generation")
    print("=" * 80)

    # ---------------------------------------------------------
    # Discover PDFs
    # ---------------------------------------------------------

    pdf_files = sorted(
        [str(pdf) for pdf in PDF_DIRECTORY.glob("*.pdf")]
    )

    if not pdf_files:

        print("\nERROR : No PDF files found in data/raw/pdfs")
        return

    print("\nLoading PDFs:\n")

    for pdf in pdf_files:

        print(f"  • {Path(pdf).name}")

    # ---------------------------------------------------------
    # Build Knowledge Base
    # ---------------------------------------------------------

    print("\nBuilding Knowledge Base...\n")

    pipeline = RAGPipeline()

    knowledge_base = pipeline.ingest(pdf_files)

    print(f"Documents : {knowledge_base.total_documents()}")
    print(f"Chunks    : {knowledge_base.total_chunks()}")
    print(f"Retriever Top-K : 5")
    print(f"LLM Provider    : {LLM_PROVIDER}")
    print(f"LLM Model       : {LLM_MODEL}")

    print("\nIndexed Documents:")

    for pdf in pdf_files:

        print(f"  - {Path(pdf).name}")

    # ---------------------------------------------------------
    # Retriever
    # ---------------------------------------------------------

    retriever = Retriever(
        vector_store=pipeline.vector_store,
        embedder=pipeline.embedder,
    )

    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------

    llm = LLMInterface(
        provider=LLM_PROVIDER,
        model_name=LLM_MODEL,
    )

    if not llm.is_available():

        print("\nERROR : LLM Provider unavailable.")
        return

    # ---------------------------------------------------------
    # Generator
    # ---------------------------------------------------------

    generator = RAGGenerator(
        retriever=retriever,
        llm=llm,
    )

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------

    verifier = VerificationPipeline()

    # ---------------------------------------------------------
    # Explainability
    # ---------------------------------------------------------

    explainer = ExplanationGenerator()

    print("\nSystem Ready.")

    while True:

        print("\n" + "=" * 80)

        question = input(
            "\nEnter Question (type 'exit' to quit): "
        )

        if question.lower() == "exit":
            break

        print("\nGenerating Answer...\n")

        # =====================================================
        # Generation
        # =====================================================

        answer = generator.generate(question)

        # =====================================================
        # Verification
        # =====================================================

        report = verifier.verify(
            answer_text=answer.text,
            retrieved_chunks=answer.retrieved_chunks,
        )

        # =====================================================
        # Explainability
        # =====================================================

        explanation = explainer.generate(
            answer,
            report,
        )

        print("\n")
        print(explanation)

        # =====================================================
        # Save Report
        # =====================================================

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        with open(
            output_dir / "latest_report.txt",
            "w",
            encoding="utf-8",
        ) as f:

            f.write(explanation)

        print("\nReport saved to outputs/latest_report.txt")

    print("\nGoodbye.")


if __name__ == "__main__":
    main()