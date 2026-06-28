from __future__ import annotations

import pickle
from pathlib import Path

from src.document.document import Document


class EmbeddingCache:
    """
    Handles persistent storage of document embeddings.

    This class is responsible only for saving and loading
    chunk embeddings. It does not generate embeddings
    or perform retrieval.
    """

    def __init__(
        self,
        cache_directory: str = "data/cache",
    ) -> None:

        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _cache_path(
        self,
        document: Document,
    ) -> Path:
        """
        Returns the cache file path for a document.
        """

        filename = f"{document.document_id}.pkl"

        return self.cache_directory / filename

    def exists(
        self,
        document: Document,
    ) -> bool:
        """
        Checks whether cached embeddings exist.
        """

        return self._cache_path(document).exists()

    def save(
        self,
        document: Document,
    ) -> None:
        """
        Saves all chunk embeddings to disk.
        """

        embeddings = [
            chunk.embedding
            for chunk in document.chunks
        ]

        with open(
            self._cache_path(document),
            "wb",
        ) as file:

            pickle.dump(
                embeddings,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(
        self,
        document: Document,
    ) -> Document:
        """
        Loads cached embeddings and restores them
        into the document chunks.
        """

        cache_path = self._cache_path(document)

        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cache found for '{document.filename}'."
            )

        with open(
            cache_path,
            "rb",
        ) as file:

            embeddings = pickle.load(file)

        if len(embeddings) != len(document.chunks):

            raise ValueError(
                "Cached embeddings do not match document chunks."
            )

        for chunk, embedding in zip(
            document.chunks,
            embeddings,
        ):
            chunk.embedding = embedding

        return document

    def delete(
        self,
        document: Document,
    ) -> None:
        """
        Deletes the cached embeddings for a document.
        """

        cache_path = self._cache_path(document)

        if cache_path.exists():
            cache_path.unlink()