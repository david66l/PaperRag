"""Reranker providers."""

from abc import ABC, abstractmethod

from app.core.logging import get_logger
from app.core.schemas import Candidate

logger = get_logger(__name__)


class BaseRerankerProvider(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        """Rerank candidates by relevance to query. Return sorted list."""
        ...


class FallbackReranker(BaseRerankerProvider):
    """No-op reranker — just uses existing fused_score."""

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        for c in candidates:
            c.rerank_score = c.fused_score
            c.final_score = c.fused_score
            c.score = c.final_score
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        logger.info("Fallback reranker: kept fused ordering for %d candidates", len(candidates))
        return candidates


class LocalRerankerProvider(BaseRerankerProvider):
    """Uses a cross-encoder model for reranking."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        self._load_model()
        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        for cand, score in zip(candidates, scores):
            cand.rerank_score = float(score)
            cand.final_score = float(score)
            cand.score = cand.final_score

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        logger.info("Local reranker: reranked %d candidates", len(candidates))
        return candidates


class APIRerankerProvider(BaseRerankerProvider):
    """Placeholder for API-based reranker (Cohere, Jina, etc.)."""

    def __init__(self, api_url: str = "", api_key: str = ""):
        self.api_url = api_url
        self.api_key = api_key

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        # TODO: implement API call
        logger.warning("API reranker not implemented, falling back to fused scores")
        return FallbackReranker().rerank(query, candidates)
