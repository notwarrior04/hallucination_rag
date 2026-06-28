from src.pipelines.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.generation.llm_interface import LLMInterface
from src.generation.rag_generator import RAGGenerator

PDF_PATH = "C:\\Users\\Sinjini Laha\\Desktop\\EHD-RAG\\data\\raw\\sample.pdf"   # Change if needed


def main():

    print("=" * 60)
    print("STEP 1 : Building Knowledge Base")
    print("=" * 60)

    pipeline = RAGPipeline()

    kb = pipeline.ingest([PDF_PATH])

    print(f"Documents : {kb.total_documents()}")
    print(f"Chunks    : {kb.total_chunks()}")

    print()

    print("=" * 60)
    print("STEP 2 : Creating Retriever")
    print("=" * 60)

    retriever = Retriever(
        vector_store=pipeline.vector_store,
        embedder=pipeline.embedder,
    )

    print("Retriever Ready")

    print()

    print("=" * 60)
    print("STEP 3 : Creating LLM")
    print("=" * 60)

    llm = LLMInterface(
        provider="gemini",
        model_name="gemini-2.5-flash",
    )

    print("Provider Available :", llm.is_available())

    print()

    print("=" * 60)
    print("STEP 4 : Creating Generator")
    print("=" * 60)

    generator = RAGGenerator(
        retriever=retriever,
        llm=llm,
    )

    print("Generator Ready")

    print()

    question = input("Question : ")

    print()

    print("=" * 60)
    print("Generating Answer...")
    print("=" * 60)

    answer = generator.generate(question)

    print()

    print("=" * 60)
    print("QUESTION")
    print("=" * 60)

    print(question)

    print()

    print("=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)

    print(answer.text)

    print()

    print("=" * 60)
    print("RETRIEVED CHUNKS")
    print("=" * 60)

    for i, chunk in enumerate(answer.retrieved_chunks, start=1):

        print(f"\n--- Chunk {i} ---")
        print(chunk.text[:500])
        print()


if __name__ == "__main__":
    main()