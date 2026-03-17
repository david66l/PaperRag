"""Abstract chunker interface."""

from abc import ABC, abstractmethod

from app.core.schemas import Chunk, PaperDocument


class BaseChunker(ABC):
    """Interface for turning PaperDocuments into Chunks."""

    @abstractmethod
    def chunk(self, doc: PaperDocument) -> list[Chunk]:
        """Return a list of Chunks from one document."""
        ...
