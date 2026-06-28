from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4
import sys
from .document import Document


class BaseParser(ABC):
    """
    Abstract base class for all document parsers.

    Every parser must implement:
        parse(filepath) -> Document

    Shared responsibilities:
    - Generate unique document IDs
    - Extract common file information
    - Initialize Document objects
    """

    def create_document(
        self,
        filepath: str,
    ) -> Document:
        """
        Creates an empty Document object with
        common metadata populated.
        """

        path = Path(filepath)

        return Document(
            document_id=str(uuid4()),
            filename=path.name,
            filetype=path.suffix.lower().replace(".", ""),
        )

    def build_metadata(
        self,
        **kwargs,
    ) -> dict:
        """
        Creates a standardized metadata dictionary.

        Can be extended later without changing
        parser implementations.
        """

        metadata = {}

        for key, value in kwargs.items():

            if value is not None:
                metadata[key] = value

        return metadata
    def validate_file(self, filepath: str) -> None:
        """
        Validate that the file exists and is readable.
        Raises appropriate exceptions if validation fails.
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"{filepath} not found.")

        if not path.is_file():
            raise ValueError(f"{filepath} is not a file.")
        
    @abstractmethod
    def parse(
        self,
        filepath: str,
    ) -> Document:
        """
        Parse the document and return a populated
        Document object.
        """
        pass