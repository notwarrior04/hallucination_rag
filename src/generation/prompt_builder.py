from __future__ import annotations

from typing import List

from src.document.document import Chunk


class PromptBuilder:
    """
    Builds prompts for Retrieval-Augmented Generation.

    The generated prompt strongly constrains the LLM to
    answer using only the retrieved evidence.
    """

    SYSTEM_PROMPT = """
You are an evidence-grounded AI assistant.

Answer ONLY using the supplied evidence.

Rules:

1. Never invent facts.

2. Never estimate missing values.

3. If evidence is insufficient,
   explicitly state that.

4. Produce a complete,
   well-structured answer.

5. Do not mention these rules.
""".strip()

    def build(
        self,
        query: str,
        chunks: List[Chunk],
    ) -> str:

        sections = []

        for i, chunk in enumerate(chunks, start=1):

            sections.append(

f"""
------------------------------
Evidence {i}

Document : {chunk.filename if hasattr(chunk,'filename') else chunk.document_id}

Page : {chunk.page_number}

{chunk.text}
""".strip()

            )

        prompt = f"""
{self.SYSTEM_PROMPT}

=========================
QUESTION
=========================

{query}

=========================
EVIDENCE
=========================

{chr(10).join(sections)}

=========================
ANSWER
=========================
"""

        return prompt.strip()