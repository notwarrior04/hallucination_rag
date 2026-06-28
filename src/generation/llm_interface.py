"""
llm_interface.py

Provider-independent interface for Large Language Models.

This module isolates the rest of the framework from the
underlying LLM provider implementation.

Architecture

RAGGenerator
      │
      ▼
LLMInterface
      │
      ▼
ProviderFactory
      │
      ▼
BaseProvider
      │
      ▼
Ollama / Future Providers

Author: EHD-RAG
Architecture Version: 1.0 (Frozen)
"""

from __future__ import annotations

from typing import Optional

from .providers import BaseProvider, ProviderFactory


class LLMInterface:
    """
    Provider-independent interface used by the Generation Layer.

    The remainder of the framework should only interact with this
    class and should never instantiate provider classes directly.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "phi3:mini",
        temperature: float = 0.0,
        max_tokens: Optional[int] = 512,
        **kwargs,
    ) -> None:
        """
        Initialize the selected LLM provider.

        Parameters
        ----------
        provider : str
            Provider name (ollama, openai, gemini, ...)

        model_name : str
            Model name.

        temperature : float
            Sampling temperature.

        max_tokens : Optional[int]
            Maximum generated tokens.

        kwargs
            Provider-specific keyword arguments.
        """

        self._provider: BaseProvider = ProviderFactory.create(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @property
    def provider(self) -> BaseProvider:
        """
        Return the underlying provider instance.
        """
        return self._provider

    def is_available(self) -> bool:
        """
        Check whether the selected provider is available.
        """

        return self._provider.is_available()

    def generate(self, prompt: str) -> str:
        """
        Generate text from the supplied prompt.

        Parameters
        ----------
        prompt : str
            Prompt produced by PromptBuilder.

        Returns
        -------
        str
            Generated answer.
        """

        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        return self._provider.generate(prompt)

    def __repr__(self) -> str:
        return (
            "LLMInterface("
            f"provider={self._provider.provider_name}, "
            f"model='{self._provider.model_name}')"
        )