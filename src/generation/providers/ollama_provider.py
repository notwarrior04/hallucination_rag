"""
ollama_provider.py

Concrete implementation of the BaseProvider interface using
a locally running Ollama server.

Requirements
------------
pip install ollama

Start Ollama:
    ollama serve

Pull a model:
    ollama pull llama3

Example
-------
provider = OllamaProvider(model_name="llama3")
answer = provider.generate(prompt)

Author: EHD-RAG
Architecture Version: 1.0 (Frozen)
"""

from __future__ import annotations

from typing import Optional

from ollama import Client

from .base_provider import BaseProvider


class GenerationError(RuntimeError):
    """Raised when text generation fails."""


class OllamaProvider(BaseProvider):
    """
    Ollama implementation of the BaseProvider interface.
    """

    def __init__(
        self,
        model_name: str = "phi3:mini",
        temperature: float = 0.0,
        max_tokens: Optional[int] = 512,
        host: str = "http://localhost:11434",
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.host = host
        self.client = Client(host=self.host)

    def is_available(self) -> bool:
        """
        Returns True if the Ollama server is reachable.
        """
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def generate(self, prompt: str) -> str:
        """
        Generate an answer from the supplied prompt.
        """

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )

            text = response["response"].strip()

            if not text:
                raise GenerationError(
                    "Ollama returned an empty response."
                )

            return text

        except Exception as exc:
            raise GenerationError(
                f"Ollama generation failed: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            "OllamaProvider("
            f"model='{self.model_name}', "
            f"host='{self.host}')"
        )