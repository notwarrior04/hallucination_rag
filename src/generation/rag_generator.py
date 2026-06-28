from __future__ import annotations

from src.generation.answer import Answer
from src.generation.prompt_builder import PromptBuilder
from src.generation.llm_interface import LLMInterface

from src.retrieval.retriever import Retriever


class RAGGenerator:
    """
    Complete Retrieval-Augmented Generation workflow.

    Responsibilities
    ----------------
    1. Retrieve evidence
    2. Build prompt
    3. Generate answer

    Post-hoc verification is NOT performed here.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMInterface,
    ) -> None:

        self.retriever = retriever
        self.prompt_builder = PromptBuilder()
        self.llm = llm

    def generate(self, query: str) -> Answer:

        print("[1] Creating Answer object")
        answer = Answer(query=query)

        print("[2] Retrieving chunks...")
        retrieved_chunks = self.retriever.retrieve(query)

        print(f"[3] Retrieved {len(retrieved_chunks)} chunks")

        answer.retrieved_chunks = retrieved_chunks

        print("[4] Building prompt...")
        answer.prompt = self.prompt_builder.build(
            query=query,
            chunks=retrieved_chunks,
        )

        print(f"[5] Prompt length: {len(answer.prompt)} characters")

        print("[6] Calling LLM...")
        generated_text = self.llm.generate(
            answer.prompt
        )

        print("[7] LLM finished")

        answer.text = generated_text

        return answer