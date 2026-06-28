from __future__ import annotations

from pathlib import Path
from typing import List

import faiss
import numpy as np

from src.document.document import Chunk, Document


class VectorStore:
    """
    FAISS-based vector index for semantic search.

    Responsibilities:
    -----------------
    - Build FAISS index
    - Save index
    - Load index
    - Perform similarity search

    Does NOT:
    ----------
    - Generate embeddings
    - Retrieve documents
    - Rank results
    """

    def __init__(
        self,
        index_directory: str = "data/vector_store",
    ) -> None:

        self.index_directory = Path(index_directory)
        self.index_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index: faiss.Index | None = None
        self.chunks: List[Chunk] = []

    def build(
        self,
        document: Document,
    ) -> None:
        """
        Builds a FAISS index from document chunks.
        """

        if not document.chunks:
            raise ValueError("Document contains no chunks.")

        embeddings = np.asarray(
            [chunk.embedding for chunk in document.chunks],
            dtype=np.float32,
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.chunks = document.chunks

    def save(
        self,
        filename: str,
    ) -> None:
        """
        Saves the FAISS index.
        """

        if self.index is None:
            raise ValueError("No FAISS index available.")

        path = self.index_directory / filename

        faiss.write_index(
            self.index,
            str(path),
        )

    def load(
        self,
        filename: str,
        chunks: List[Chunk],
    ) -> None:
        """
        Loads an existing FAISS index.
        """

        path = self.index_directory / filename

        if not path.exists():
            raise FileNotFoundError(path)

        self.index = faiss.read_index(str(path))

        self.chunks = chunks

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Chunk]:
        """
        Returns the most similar chunks.
        """

        if self.index is None:
            raise ValueError("Vector index has not been built.")

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        ).reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for idx in indices[0]:

            if idx == -1:
                continue

            results.append(
                self.chunks[idx]
            )

        return results