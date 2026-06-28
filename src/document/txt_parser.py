from uuid import uuid4

from .document import Document


class TXTParser:

    def parse(self, filepath):

        with open(filepath, "r", encoding="utf-8") as f:

            text = f.read()

        return Document(

            document_id=str(uuid4()),

            filename=filepath.split("/")[-1],

            filetype="txt",

            raw_text=text,

            total_pages=1

        )