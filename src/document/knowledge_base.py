from __future__ import annotations

from typing import List

from .document import Chunk, Document


class KnowledgeBase:
    """
    Represents the complete collection of uploaded
    documents for a single RAG session.

    Provides helper methods for managing documents
    and accessing all searchable chunks.
    """

    def __init__(self) -> None:

        self.documents: List[Document] = []

    def add_document(
        self,
        document: Document,
    ) -> None:
        """
        Add a document to the knowledge base.
        """
        self.documents.append(document)

    def remove_document(
        self,
        document_id: str,
    ) -> None:
        """
        Remove a document by its ID.
        """
        self.documents = [
            doc
            for doc in self.documents
            if doc.document_id != document_id
        ]

    def get_document(
        self,
        document_id: str,
    ) -> Document | None:
        """
        Retrieve a document by its ID.
        """
        for document in self.documents:

            if document.document_id == document_id:
                return document

        return None

    def list_documents(
        self,
    ) -> List[Document]:
        """
        Returns all uploaded documents.
        """
        return self.documents

    def total_documents(
        self,
    ) -> int:
        """
        Returns the total number of uploaded documents.
        """
        return len(self.documents)

    def clear(
        self,
    ) -> None:
        """
        Removes every document.
        """
        self.documents.clear()

    # -----------------------------------------------------
    # Retrieval Helpers
    # -----------------------------------------------------

    def all_chunks(
        self,
    ) -> List[Chunk]:
        """
        Returns every chunk from every document.
        """

        chunks: List[Chunk] = []

        for document in self.documents:
            chunks.extend(document.chunks)

        return chunks

    def total_chunks(
        self,
    ) -> int:
        """
        Returns the total number of chunks.
        """
        return len(self.all_chunks())

    def to_document(
        self,
    ) -> Document:
        """
        Creates a virtual document containing all chunks.

        Used internally for building a unified vector index.
        """

        merged = Document(
            document_id="knowledge_base",
            filename="knowledge_base",
            filetype="virtual",
        )

        merged.chunks = self.all_chunks()

        return merged