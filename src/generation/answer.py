from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.document.document import Chunk


@dataclass
class Answer:
    """
    Represents the complete answer produced by the Explainable
    Hybrid RAG framework.

    Every stage of the pipeline enriches this object instead
    of creating new data structures.

    Query
        ↓

    Retrieval
        ↓

    Generation
        ↓

    Verification
        ↓

    Explainability
        ↓

    Rendering
    """

    # -------------------------------------------------
    # User Query
    # -------------------------------------------------

    query: str

    # -------------------------------------------------
    # Prompt given to LLM
    # -------------------------------------------------

    prompt: str = ""

    # -------------------------------------------------
    # Final generated answer
    # -------------------------------------------------

    text: str = ""

    # -------------------------------------------------
    # Evidence used during retrieval
    # -------------------------------------------------

    retrieved_chunks: List[Chunk] = field(default_factory=list)

    # -------------------------------------------------
    # Verification (filled later)
    # -------------------------------------------------

    claims: List[Any] = field(default_factory=list)

    # -------------------------------------------------
    # Overall confidence
    # -------------------------------------------------

    confidence: float | None = None

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    retrieval_time: float | None = None

    generation_time: float | None = None

    verification_time: float | None = None

    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    metadata: Dict[str, Any] = field(default_factory=dict)