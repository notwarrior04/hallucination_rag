import re
from typing import List, Dict

import spacy


class ClaimExtractor:
    """
    Hybrid claim extractor for EHD-RAG.

    Phase 1:
    - Sentence segmentation
    - Factual sentence filtering

    Future:
    - LLM-assisted decomposition
    - Atomic claim splitting
    """

    def __init__(self):

        self.nlp = spacy.load("en_core_web_sm")

    def _is_factual(
            self,
            sentence: str
        ):

            sentence = sentence.strip()

            if not sentence:
                return False

            if sentence.endswith("?"):
                return False

            return True
            
    def extract_claims(
        self,
        text: str
    ) -> List[Dict]:

        doc = self.nlp(text)

        claims = []

        sentences = [
            sent.text.strip()
            for sent in doc.sents
        ]

        # Fallback for very short answers like:
        # "2003"
        # "Paris"
        # "Yes"

        if not sentences:

            text = text.strip()

            if text:

                return [
                    {
                        "claim_id": 1,
                        "claim": text
                    }
                ]

            return []

        claim_id = 1

        for sentence in sentences:

            sentence = re.sub(
                r"\s+",
                " ",
                sentence
            )

            if not self._is_factual(sentence):
                continue

            claims.append(
                {
                    "claim_id": claim_id,
                    "claim": sentence
                }
            )

            claim_id += 1

        return claims
