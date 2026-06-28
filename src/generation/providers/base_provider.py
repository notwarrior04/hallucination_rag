"""
base_provider.py

Defines the abstract interface that every LLM provider must implement.

This module is intentionally provider-agnostic.

Supported providers (current/future):
    - Ollama
    - OpenAI
    - Gemini
    - LM Studio
    - vLLM

Architecture:
    RAGGenerator
          │
          ▼
    LLMInterface
          │
          ▼
    BaseProvider
          │
     Provider Implementation

Author: EHD-RAG
Architecture Version: 1.0 (Frozen)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseProvider(ABC):
    """
    Abstract base class for every LLM provider.

    Every provider must expose exactly one public generation
    interface so that the rest of the framework remains
    provider-independent.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        """
        Initialize the provider.

        Parameters
        ----------
        model_name : str
            Name of the model.

        temperature : float
            Sampling temperature.

        max_tokens : Optional[int]
            Maximum tokens to generate.
        """

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response for the supplied prompt.

        Parameters
        ----------
        prompt : str
            Fully constructed prompt.

        Returns
        -------
        str
            Generated answer.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check whether the provider is available.

        Examples
        --------
        - Ollama server running
        - API key valid
        - Local model installed

        Returns
        -------
        bool
        """
        raise NotImplementedError

    @property
    def provider_name(self) -> str:
        """
        Human-readable provider name.
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        return (
            f"{self.provider_name}("
            f"model='{self.model_name}', "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )