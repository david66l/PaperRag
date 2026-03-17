"""Reciprocal Rank Fusion (RRF)."""

from collections import defaultdict

from app.core.schemas import Candidate
from app.retrieval.fusion.base import BaseFusionStrategy


class RRFFusion(BaseFusionStrategy):
    """Fuse using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i) over all ranking lists.
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, candidates: list[Candidate]) -> list[Candidate]:
        # Build per-signal rankings
        rankings: dict[str, list[Candidate]] = {
            "dense": sorted(candidates, key=lambda c: c.source_scores.dense_score, reverse=True),
            "bm25": sorted(candidates, key=lambda c: c.source_scores.bm25_score, reverse=True),
            "meta": sorted(candidates, key=lambda c: c.source_scores.metadata_score, reverse=True),
        }

        rrf_scores: dict[str, float] = defaultdict(float)
        for _signal, ranked in rankings.items():
            for rank, cand in enumerate(ranked, start=1):
                rrf_scores[cand.chunk_id] += 1.0 / (self.k + rank)

        # Map back
        cand_map = {c.chunk_id: c for c in candidates}
        for chunk_id, score in rrf_scores.items():
            if chunk_id in cand_map:
                cand_map[chunk_id].fused_score = score

        result = list(cand_map.values())
        result.sort(key=lambda c: c.fused_score, reverse=True)
        return result
