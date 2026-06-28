from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.document.knowledge_base import KnowledgeBase
from src.document.document_loader import DocumentLoader
from src.document.cleaner import DocumentCleaner
from src.document.chunker import DocumentChunker

from src.retrieval.embedding_generator import EmbeddingGenerator
from src.retrieval.embedding_cache import EmbeddingCache
from src.retrieval.vector_store import VectorStore


class RAGPipeline:
    """
    Complete ingestion pipeline for the Explainable Hybrid RAG system.

    Responsibilities
    ----------------
    - Load uploaded documents
    - Clean extracted text
    - Chunk documents
    - Generate embeddings
    - Cache embeddings
    - Build the vector index

    This class is responsible ONLY for document ingestion.

    It does NOT answer questions.
    """

    def __init__(self) -> None:

        self.loader = DocumentLoader()

        self.cleaner = DocumentCleaner()

        self.chunker = DocumentChunker()

        self.embedder = EmbeddingGenerator()

        self.cache = EmbeddingCache()

        self.vector_store = VectorStore()

    def ingest(
        self,
        filepaths: Iterable[str | Path],
    ) -> KnowledgeBase:

        knowledge_base = KnowledgeBase()

        for filepath in filepaths:

            filepath = str(filepath)

            document = self.loader.load(filepath)

            document = self.cleaner.clean(document)

            document = self.chunker.chunk(document)

            if self.cache.exists(document):

                document = self.cache.load(document)

            else:

                document = self.embedder.generate(document)

                self.cache.save(document)

            knowledge_base.add_document(document)

        merged_document = knowledge_base.to_document()

        self.vector_store.build(merged_document)

        return knowledge_base