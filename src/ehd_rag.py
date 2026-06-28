from src.generation.rag_generator import RAGGenerator
from src.verification.verification_pipeline import VerificationPipeline
from src.explainability.explanation_generator import ExplanationGenerator


class EHDRAG:

    def __init__(
        self,
        rag_generator,
    ):

        self.rag = rag_generator

        self.verifier = VerificationPipeline()

        self.explainer = ExplanationGenerator()

    def answer(
        self,
        question,
    ):

        answer = self.rag.generate(question)

        report = self.verifier.verify(
            answer.text,
            answer.retrieved_chunks,
        )

        explanation = self.explainer.generate(
            answer,
            report,
        )

        return answer, report, explanation