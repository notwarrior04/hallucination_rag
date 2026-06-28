"""
gemini_provider.py

Google Gemini implementation of the BaseProvider interface.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

from .base_provider import BaseProvider


class GeminiProvider(BaseProvider):

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_tokens: int | None = 512,
    ) -> None:

        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found in .env"
            )

        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text

    def is_available(self) -> bool:

        try:

            self.client.models.generate_content(
                model=self.model_name,
                contents="Hello",
            )

            return True

        except Exception:

            return False