from __future__ import annotations

from typing import List

import numpy as np

from src.document.document import Chunk
from src.retrieval.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import VectorStore


class Retriever:
    """
    Dense semantic retriever.

    Responsibilities
    ----------------
    - Encode user query
    - Search vector index
    - Return the most relevant chunks

    This class is intentionally independent of
    any LLM implementation.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingGenerator,
    ) -> None:

        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Chunk]:
        """
        Retrieve the most relevant chunks for a query.
        """

        query_embedding = self.embedder.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        return self.vector_store.search(
            query_embedding,
            top_k,
        )