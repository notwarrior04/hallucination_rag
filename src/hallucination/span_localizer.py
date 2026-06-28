import re
import spacy


class SpanLocalizer:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text):

        doc = self.nlp(text)

        entities = []

        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_
                }
            )

        return entities

    def localize(
        self,
        claim,
        evidence
    ):

        claim_entities = self.extract_entities(
            claim
        )

        evidence_entities = self.extract_entities(
            evidence
        )

        mismatches = []

        # DATE MISMATCH

        claim_years = re.findall(
            r"\b\d{4}\b",
            claim
        )

        evidence_years = re.findall(
            r"\b\d{4}\b",
            evidence
        )

        if claim_years and evidence_years:

            if claim_years[0] != evidence_years[0]:

                mismatches.append(
                    {
                        "type": "DATE_MISMATCH",
                        "claim": claim_years[0],
                        "evidence": evidence_years[0]
                    }
                )

        return {
            "claim_entities":
                claim_entities,

            "evidence_entities":
                evidence_entities,

            "mismatches":
                mismatches
        }