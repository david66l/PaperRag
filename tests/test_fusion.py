"""Tests for retrieval fusion strategies."""

# OPTIMIZED_BY_CODEX_STEP_4
from app.core.schemas import Candidate, SourceScores
from app.retrieval.fusion.rrf import RRFFusion
from app.retrieval.fusion.weighted import WeightedFusion


def _candidate(chunk_id: str, dense: float, bm25: float, meta: float) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        text=f"text-{chunk_id}",
        source_type="metadata",
        source_scores=SourceScores(dense_score=dense, bm25_score=bm25, metadata_score=meta),
    )


def test_weighted_fusion_sorts_by_combined_score():
    candidates = [
        _candidate("a", dense=0.9, bm25=0.1, meta=0.1),
        _candidate("b", dense=0.2, bm25=0.9, meta=0.1),
        _candidate("c", dense=0.1, bm25=0.2, meta=0.9),
    ]

    fused = WeightedFusion(dense_w=0.5, bm25_w=0.3, meta_w=0.2).fuse(candidates)

    assert len(fused) == 3
    assert fused[0].fused_score >= fused[1].fused_score >= fused[2].fused_score


def test_rrf_fusion_assigns_scores_for_all_candidates():
    candidates = [
        _candidate("x", dense=0.5, bm25=0.4, meta=0.3),
        _candidate("y", dense=0.4, bm25=0.5, meta=0.2),
        _candidate("z", dense=0.3, bm25=0.2, meta=0.9),
    ]

    fused = RRFFusion(k=10).fuse(candidates)

    assert len(fused) == 3
    assert all(c.fused_score > 0 for c in fused)
    assert fused[0].fused_score >= fused[-1].fused_score


# STEP_4_SUMMARY: Added fusion tests covering weighted and reciprocal-rank strategies.
