"""Abstract base loader."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from app.core.schemas import RawPaperRecord


class BaseLoader(ABC):
    """Interface for data loaders."""

    @abstractmethod
    def load(self, path: Path, limit: int = 0) -> Iterator[RawPaperRecord]:
        """Yield raw paper records from *path*.

        Args:
            path: File path to load.
            limit: Max records to yield. 0 = unlimited.
        """
        ...
