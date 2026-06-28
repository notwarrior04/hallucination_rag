from pathlib import Path

from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .txt_parser import TXTParser
from .pptx_parser import PPTXParser


class DocumentLoader:

    def __init__(self):

        self.parsers = {

            ".pdf": PDFParser(),

            ".docx": DOCXParser(),

            ".txt": TXTParser(),

            ".pptx": PPTXParser()

        }

    def load(self, filepath):

        extension = Path(filepath).suffix.lower()

        parser = self.parsers.get(extension)

        if parser is None:

            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        return parser.parse(filepath)