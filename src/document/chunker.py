from uuid import uuid4

from .document import Chunk, Document


class DocumentChunker:
    """
    Splits a cleaned document into paragraph-based chunks.
    """
    def chunk(self, document: Document) -> Document:
        chunks = []
        chunk_index = 0
        for page in document.pages:
            paragraphs = [
                paragraph.strip()
                for paragraph in page.text.split("\n\n")
                if paragraph.strip()
            ]
            for paragraph_number, paragraph in enumerate(
                paragraphs,
                start=1
            ):
                chunk = Chunk(
                    chunk_id=str(uuid4()),
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    text=paragraph,
                    page_number=page.page_number,
                    paragraph_number=paragraph_number,
                    section=None,
                    start_char=None,
                    end_char=None,
                    embedding=None,
                    metadata={}
                )
                chunks.append(chunk)
                chunk_index += 1
        document.chunks = chunks
        return document