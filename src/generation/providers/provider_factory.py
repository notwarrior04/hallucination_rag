"""
provider_factory.py

Factory responsible for creating LLM provider instances.

The rest of the framework should NEVER directly instantiate
provider classes. All provider creation must go through this
factory.

Architecture:

LLMInterface
      │
      ▼
ProviderFactory
      │
 ┌────┼─────┐
 ▼    ▼     ▼
Ollama Gemini OpenAI

Author: EHD-RAG
Architecture Version: 1.0 (Frozen)
"""

from __future__ import annotations

from typing import Optional

from .gemini_provider import GeminiProvider
from .base_provider import BaseProvider
from .ollama_provider import OllamaProvider


class ProviderFactory:
    """
    Factory for constructing LLM providers.

    Future providers can be added without modifying
    the rest of the framework.
    """

    @staticmethod
    def create(
        provider: str = "ollama",
        model_name: str = "phi3:mini",
        temperature: float = 0.0,
        max_tokens: Optional[int] = 512,
        **kwargs,
    ) -> BaseProvider:
        """
        Create a provider instance.

        Parameters
        ----------
        provider : str
            Provider name.

        model_name : str
            Model to use.

        temperature : float
            Sampling temperature.

        max_tokens : Optional[int]
            Maximum number of generated tokens.

        Returns
        -------
        BaseProvider
        """

        provider = provider.lower().strip()

        if provider == "ollama":
            return OllamaProvider(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                host=kwargs.get("host", "http://localhost:11434"),
            )
        
        if provider == "gemini":
            return GeminiProvider(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        raise ValueError(
            f"Unsupported LLM provider: '{provider}'."
        )

    @staticmethod
    def available_providers() -> list[str]:
        """
        Return the list of currently supported providers.
        """

        return [
            "ollama",
            "gemini",
        ]