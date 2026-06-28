from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ==========================================================
# PAGE
# Represents a single page within a document.
# ==========================================================
@dataclass
class Page:
    page_number: int
    text: str
    tables: List[Any] = field(default_factory=list)
    images: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

# ==========================================================
# CHUNK
# Smallest searchable semantic unit.
# ==========================================================
@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    section: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# ==========================================================
# DOCUMENT
# Represents one uploaded document.
# ==========================================================
@dataclass
class Document:
    document_id: str
    filename: str
    filetype: str

    # -----------------------------------------------------------------
    # Temporary compatibility field.
    # Existing cleaner/chunker currently operate on raw_text.
    # Eventually this will be generated from pages automatically.
    # -----------------------------------------------------------------
    raw_text: str = ""

    # -----------------------------------------------------------------
    # Permanent document representation.
    # -----------------------------------------------------------------
    pages: List[Page] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Chunks generated from this document.
    # -----------------------------------------------------------------
    chunks: List[Chunk] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Document metadata
    # Examples:
    # author
    # creation_date
    # language
    # title
    # subject
    # OCR confidence
    # -----------------------------------------------------------------
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_pages(self) -> int:
        """
        Returns the total number of pages.
        """
        return len(self.pages)