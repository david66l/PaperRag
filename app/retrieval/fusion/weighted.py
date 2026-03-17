"""Weighted sum fusion strategy."""

from app.core.schemas import Candidate
from app.retrieval.fusion.base import BaseFusionStrategy
from app.retrieval.fusion.normalizer import ScoreNormalizer


class WeightedFusion(BaseFusionStrategy):
    """Fuse via weighted sum of normalized scores."""

    def __init__(self, dense_w: float = 0.6, bm25_w: float = 0.3, meta_w: float = 0.1):
        self.dense_w = dense_w
        self.bm25_w = bm25_w
        self.meta_w = meta_w

    def fuse(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates = ScoreNormalizer.normalize_all(candidates)

        for c in candidates:
            c.fused_score = (
                self.dense_w * c.source_scores.dense_score
                + self.bm25_w * c.source_scores.bm25_score
                + self.meta_w * c.source_scores.metadata_score
            )

        candidates.sort(key=lambda c: c.fused_score, reverse=True)
        return candidates
