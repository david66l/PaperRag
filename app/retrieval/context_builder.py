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
        selected = self._reorder_by_source(selected)
        context_parts: list[str] = []
        citations: list[Citation] = []

        for i, cand in enumerate(selected, 1):
            block = self._format_block(i, cand)
            context_parts.append(block)

            citations.append(
                Citation(
                    chunk_id=cand.chunk_id,
                    doc_id=cand.doc_id,
                    source_type=cand.source_type,
                    title=cand.title,
                    paper_id=cand.paper_id,
                    authors=cand.authors,
                    categories=cand.categories,
                    published=cand.published,
                    file_name=cand.file_name,
                    file_path=cand.file_path,
                    page_no=cand.page_no,
                    relevance_score=cand.final_score,
                )
            )

        context_text = "\n---\n".join(context_parts)
        logger.info("Context built: %d chunks, ~%d chars", len(selected), len(context_text))
        return context_text, citations

    def _select(self, candidates: list[Candidate]) -> list[Candidate]:
        """Pick top_n candidates while respecting a rough token budget."""
        selected: list[Candidate] = []
        total_chars = 0

        for cand in candidates:
            if len(selected) >= self.top_n:
                break
            # Rough token budget check (1 token ≈ 4 chars)
            if total_chars + len(cand.text) > self.max_tokens * 4:
                break
            selected.append(cand)
            total_chars += len(cand.text)

        return selected

    @staticmethod
    def _reorder_by_source(candidates: list[Candidate]) -> list[Candidate]:
        pdf_chunks = [c for c in candidates if c.source_type == "pdf"]
        metadata_chunks = [c for c in candidates if c.source_type == "metadata"]
        return pdf_chunks + metadata_chunks

    @staticmethod
    def _format_block(i: int, cand: Candidate) -> str:
        if cand.source_type == "pdf":
            return (
                f"[{i}] Source: pdf\n"
                f"    File: {cand.file_name or ''}\n"
                f"    Path: {cand.file_path or ''}\n"
                f"    Page: {cand.page_no or ''}\n"
                f"    Content: {cand.text}\n"
            )

        authors_str = ", ".join(cand.authors[:3])
        if len(cand.authors) > 3:
            authors_str += " et al."
        cats_str = ", ".join(cand.categories[:3])
        return (
            f"[{i}] Source: metadata\n"
            f"    Title: {cand.title or ''}\n"
            f"    Paper ID: {cand.paper_id or cand.doc_id}\n"
            f"    Authors: {authors_str}\n"
            f"    Categories: {cats_str}\n"
            f"    Published: {cand.published or ''}\n"
            f"    Content: {cand.text}\n"
        )
