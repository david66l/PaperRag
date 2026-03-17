"""Abstract embedding provider."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingProvider(ABC):
    """Interface for embedding providers."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of document texts.

        Returns:
            np.ndarray of shape (len(texts), dim), dtype float32.
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query.

        Returns:
            np.ndarray of shape (dim,), dtype float32.
        """
        ...
