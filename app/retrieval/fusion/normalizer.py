"""Score normalization strategies."""

from app.core.schemas import Candidate


class ScoreNormalizer:
    """Normalize heterogeneous scores to [0, 1]."""

    @staticmethod
    def min_max(candidates: list[Candidate], field: str) -> list[Candidate]:
        """Min-max normalize a specific score field across candidates."""
        if not candidates:
            return candidates

        scores = [getattr(c.source_scores, field) for c in candidates]
        lo, hi = min(scores), max(scores)
        rng = hi - lo if hi > lo else 1.0

        for c in candidates:
            raw = getattr(c.source_scores, field)
            setattr(c.source_scores, field, (raw - lo) / rng)

        return candidates

    @staticmethod
    def normalize_all(candidates: list[Candidate]) -> list[Candidate]:
        """Normalize all source score fields."""
        for field in ("dense_score", "bm25_score", "metadata_score"):
            ScoreNormalizer.min_max(candidates, field)
        return candidates
