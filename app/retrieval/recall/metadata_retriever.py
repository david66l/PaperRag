"""Metadata-boost retrieval — lightweight rule-based scoring."""

import re

from app.core.logging import get_logger
from app.core.schemas import Candidate, SourceScores
from app.storage.repositories.chunk_repository import ChunkRepository

logger = get_logger(__name__)

# Common AI/ML terms that signal specific topic intent
BOOST_TERMS = {
    "lora", "qlora", "adapter", "transformer", "bert", "gpt", "llm",
    "rag", "retrieval", "rlhf", "ppo", "dpo", "instruction tuning",
    "fine-tuning", "finetuning", "diffusion", "attention", "mamba",
    "mixture of experts", "moe", "chain of thought", "cot",
    "reinforcement learning", "contrastive learning", "knowledge distillation",
}

YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")


class MetadataRetriever:
    """Boost candidates based on metadata matching (categories, year, terms)."""

    def __init__(self, chunk_repo: ChunkRepository):
        self.chunk_repo = chunk_repo

    def boost(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        """Add metadata_score to existing candidates."""
        query_lower = query.lower()
        query_years = set(YEAR_PATTERN.findall(query))
        query_terms = [t for t in BOOST_TERMS if t in query_lower]

        for cand in candidates:
            score = 0.0

            # Year match
            if query_years:
                for year in query_years:
                    if year in cand.update_date:
                        score += 0.3

            # Term match in title
            title_lower = cand.title.lower()
            for term in query_terms:
                if term in title_lower:
                    score += 0.2

            # Category match
            cat_str = " ".join(cand.categories).lower()
            if "cs.cl" in cat_str and any(t in query_lower for t in ["nlp", "language", "text"]):
                score += 0.1
            if "cs.lg" in cat_str and any(t in query_lower for t in ["learning", "training", "model"]):
                score += 0.1

            cand.source_scores.metadata_score = min(score, 1.0)

        boosted = sum(1 for c in candidates if c.source_scores.metadata_score > 0)
        logger.info("Metadata boost: %d/%d candidates boosted", boosted, len(candidates))
        return candidates
