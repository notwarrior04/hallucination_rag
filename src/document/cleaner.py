import re

from .document import Document


class DocumentCleaner:
    """
    Cleans extracted document text while preserving
    the document structure.
    """

    def _clean_text(self, text: str) -> str:

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        text = "\n".join(
            line.rstrip()
            for line in text.split("\n")
        )

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"\n\s*\n+", "\n\n", text)

        return text.strip()

    def clean(self, document: Document) -> Document:
        for page in document.pages:
            page.text = self._clean_text(page.text)
        document.raw_text = "\n".join(page.text  for page in document.pages)
        return document