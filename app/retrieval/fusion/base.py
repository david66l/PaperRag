"""Abstract fusion strategy."""

from abc import ABC, abstractmethod

from app.core.schemas import Candidate


class BaseFusionStrategy(ABC):
    @abstractmethod
    def fuse(self, candidates: list[Candidate]) -> list[Candidate]:
        """Fuse scores and sort candidates. Return sorted list."""
        ...
