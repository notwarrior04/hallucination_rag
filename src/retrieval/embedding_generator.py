from sentence_transformers import SentenceTransformer

from src.document.document import Document


class EmbeddingGenerator:
    """
    Generates embeddings for every chunk
    inside a document.
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):
        self.model = SentenceTransformer(model_name)

    def generate(
        self,
        document: Document
    ) -> Document:

        if not document.chunks:
            raise ValueError(
                "Document contains no chunks."
            )

        texts = [

            chunk.text

            for chunk in document.chunks

        ]

        embeddings = self.model.encode(

            texts,

            normalize_embeddings=True,

            show_progress_bar=True,

            convert_to_numpy=True

        )

        for chunk, embedding in zip(
            document.chunks,
            embeddings
        ):

            chunk.embedding = embedding.tolist()

        return document