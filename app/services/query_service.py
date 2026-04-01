"""Query service — orchestrates retrieval + generation."""

import time
from typing import Literal

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.schemas import Citation, QueryResponse, RetrievalResult
from app.embedding.pipeline import create_embedding_provider
from app.generation.pipeline import GenerationPipeline
from app.retrieval.pipeline import RetrievalPipeline
from app.storage.persistence import PersistenceManager

logger = get_logger(__name__)


class QueryService:
    """Handles user queries end-to-end."""

    def __init__(self, settings: Settings, persistence: PersistenceManager):
        self.settings = settings
        self.persistence = persistence
        self._retrieval: RetrievalPipeline | None = None
        self._generation: GenerationPipeline | None = None

    def _ensure_pipelines(self) -> None:
        if self._retrieval is None:
            provider = create_embedding_provider(self.settings)
            self._retrieval = RetrievalPipeline(self.settings, self.persistence, provider)
        if self._generation is None:
            self._generation = GenerationPipeline(self.settings)

    def query(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["concise", "analysis"] = "concise",
    ) -> QueryResponse:
        start = time.time()
        self._ensure_pipelines()

        # 1. Retrieval
        retrieval_result: RetrievalResult = self._retrieval.run(query, top_k=top_k)

        # 2. Generation
        answer, citations = self._generation.run(retrieval_result, mode=mode)
        evidence_level = self._infer_evidence_level(citations)

        elapsed = (time.time() - start) * 1000
        logger.info("Query answered in %.1f ms", elapsed)

        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieval_result.candidates,
            evidence_level=evidence_level,
            retrieval_trace=retrieval_result.trace,
            elapsed_ms=elapsed,
        )

    @staticmethod
    def _infer_evidence_level(citations: list[Citation]) -> str:
        source_types = {c.source_type for c in citations}
        if not source_types:
            return "metadata"
        if source_types == {"metadata"}:
            return "metadata"
        if source_types == {"pdf"}:
            return "pdf"
        return "hybrid"
