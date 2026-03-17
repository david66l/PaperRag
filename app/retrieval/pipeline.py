"""Retrieval pipeline: recall → merge → fusion → rerank → context build."""

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.schemas import Candidate, RetrievalResult
from app.embedding.providers.base import BaseEmbeddingProvider
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.fusion.base import BaseFusionStrategy
from app.retrieval.fusion.rrf import RRFFusion
from app.retrieval.fusion.weighted import WeightedFusion
from app.retrieval.ranking.reranker import (
    BaseRerankerProvider,
    FallbackReranker,
    LocalRerankerProvider,
)
from app.retrieval.recall.bm25_retriever import BM25Retriever
from app.retrieval.recall.dense_retriever import DenseRetriever
from app.retrieval.recall.metadata_retriever import MetadataRetriever
from app.storage.persistence import PersistenceManager

logger = get_logger(__name__)


def _create_fusion(settings: Settings) -> BaseFusionStrategy:
    if settings.fusion_strategy == "weighted":
        return WeightedFusion(settings.dense_weight, settings.bm25_weight, settings.metadata_weight)
    return RRFFusion(k=settings.rrf_k)


def _create_reranker(settings: Settings) -> BaseRerankerProvider:
    if not settings.rerank_enabled or settings.reranker_provider == "none":
        return FallbackReranker()
    if settings.reranker_provider == "local":
        return LocalRerankerProvider(model_name=settings.reranker_model)
    return FallbackReranker()


class RetrievalPipeline:
    """Orchestrates multi-path recall → fusion → rerank → context building."""

    def __init__(
        self,
        settings: Settings,
        persistence: PersistenceManager,
        embedding_provider: BaseEmbeddingProvider,
    ):
        self.settings = settings
        self.dense_retriever = DenseRetriever(
            persistence.vector_repo, persistence.chunk_repo, embedding_provider
        )
        self.bm25_retriever = BM25Retriever(
            persistence.keyword_repo, persistence.chunk_repo
        )
        self.metadata_retriever = MetadataRetriever(persistence.chunk_repo)
        self.fusion: BaseFusionStrategy = _create_fusion(settings)
        self.reranker: BaseRerankerProvider = _create_reranker(settings)
        self.context_builder = ContextBuilder(
            top_n=settings.top_n_context,
            max_tokens=settings.context_max_tokens,
        )

    def run(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Execute the full retrieval pipeline."""
        top_k_dense = top_k or self.settings.top_k_dense
        top_k_bm25 = top_k or self.settings.top_k_bm25

        # 1. Multi-path recall
        dense_cands = self.dense_retriever.retrieve(query, top_k_dense)
        bm25_cands = self.bm25_retriever.retrieve(query, top_k_bm25)

        # 2. Merge candidates (deduplicate by chunk_id, combine scores)
        merged = self._merge(dense_cands, bm25_cands)

        # 3. Metadata boost
        merged = self.metadata_retriever.boost(query, merged)

        # 4. Fusion
        fused = self.fusion.fuse(merged)
        fused = fused[: self.settings.top_n_fused]

        # 5. Rerank
        reranked = self.reranker.rerank(query, fused)
        reranked = reranked[: self.settings.top_n_final]

        # 6. Build context
        context_text, citations = self.context_builder.build(reranked)

        trace = {
            "dense_count": len(dense_cands),
            "bm25_count": len(bm25_cands),
            "merged_count": len(merged),
            "fused_count": len(fused),
            "reranked_count": len(reranked),
            "context_chunks": len(citations),
        }
        logger.info("Retrieval trace: %s", trace)

        return RetrievalResult(
            query=query,
            candidates=reranked,
            context_text=context_text,
            citations=citations,
            trace=trace,
        )

    @staticmethod
    def _merge(dense: list[Candidate], bm25: list[Candidate]) -> list[Candidate]:
        """Merge two candidate lists, combining scores for duplicates."""
        merged: dict[str, Candidate] = {}

        for c in dense:
            merged[c.chunk_id] = c

        for c in bm25:
            if c.chunk_id in merged:
                merged[c.chunk_id].source_scores.bm25_score = c.source_scores.bm25_score
            else:
                merged[c.chunk_id] = c

        return list(merged.values())
