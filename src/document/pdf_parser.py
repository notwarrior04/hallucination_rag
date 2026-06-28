import pymupdf as fitz

from .base_parser import BaseParser
from .document import Document, Page


class PDFParser(BaseParser):

    def parse(
        self,
        filepath: str,
    ) -> Document:

        self.validate_file(filepath)

        pdf = fitz.open(filepath)

        document = self.create_document(filepath)

        # -----------------------------------------
        # PDF Metadata
        # -----------------------------------------

        meta = pdf.metadata

        document.metadata = self.build_metadata(

            title=meta.get("title"),

            author=meta.get("author"),

            subject=meta.get("subject"),

            creator=meta.get("creator"),

            producer=meta.get("producer"),

            creation_date=meta.get("creationDate"),

            modification_date=meta.get("modDate")

        )

        # -----------------------------------------
        # Extract Pages
        # -----------------------------------------

        full_text = []

        for page_index in range(len(pdf)):

            page = pdf.load_page(page_index)

            text = page.get_text("text")

            document.pages.append(

                Page(

                    page_number=page_index + 1,

                    text=text

                )

            )

            full_text.append(text)

        # Temporary compatibility
        document.raw_text = "\n".join(full_text)

        pdf.close()

        return document