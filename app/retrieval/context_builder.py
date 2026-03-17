"""Build the final context string + citations from ranked candidates."""

from app.core.logging import get_logger
from app.core.schemas import Candidate, Citation

logger = get_logger(__name__)


class ContextBuilder:
    """Selects top candidates and formats them for the generation module."""

    def __init__(self, top_n: int = 5, max_tokens: int = 3000):
        self.top_n = top_n
        self.max_tokens = max_tokens

    def build(self, candidates: list[Candidate]) -> tuple[str, list[Citation]]:
        """Return (formatted_context, citations)."""
        selected = self._select(candidates)
        context_parts: list[str] = []
        citations: list[Citation] = []

        for i, cand in enumerate(selected, 1):
            authors_str = ", ".join(cand.authors[:3])
            if len(cand.authors) > 3:
                authors_str += " et al."
            cats_str = ", ".join(cand.categories[:3])

            block = (
                f"[{i}] Title: {cand.title}\n"
                f"    Authors: {authors_str}\n"
                f"    Categories: {cats_str}\n"
                f"    Date: {cand.update_date}\n"
                f"    Content: {cand.text}\n"
            )
            context_parts.append(block)

            citations.append(
                Citation(
                    chunk_id=cand.chunk_id,
                    doc_id=cand.doc_id,
                    title=cand.title,
                    authors=cand.authors,
                    categories=cand.categories,
                    update_date=cand.update_date,
                    relevance_score=cand.final_score,
                )
            )

        context_text = "\n---\n".join(context_parts)
        logger.info("Context built: %d chunks, ~%d chars", len(selected), len(context_text))
        return context_text, citations

    def _select(self, candidates: list[Candidate]) -> list[Candidate]:
        """Pick top_n while deduplicating by doc_id and respecting token budget."""
        seen_docs: set[str] = set()
        selected: list[Candidate] = []
        total_chars = 0

        for cand in candidates:
            if len(selected) >= self.top_n:
                break
            # Deduplicate by doc_id (prefer highest-scored chunk per paper)
            if cand.doc_id in seen_docs:
                continue
            # Rough token budget check (1 token ≈ 4 chars)
            if total_chars + len(cand.text) > self.max_tokens * 4:
                break
            seen_docs.add(cand.doc_id)
            selected.append(cand)
            total_chars += len(cand.text)

        return selected
