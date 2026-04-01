"""Format citations for display."""

from app.core.schemas import Citation


class CitationFormatter:
    """Format citations into a readable reference list."""

    @staticmethod
    def format_references(citations: list[Citation]) -> str:
        """Generate a numbered reference list."""
        lines = []
        for i, c in enumerate(citations, 1):
            if c.source_type == "pdf":
                line = f"[{i}] PDF: {c.file_name or c.doc_id} (page {c.page_no or '?'})"
            else:
                authors = ", ".join(c.authors[:3])
                if len(c.authors) > 3:
                    authors += " et al."
                cats = ", ".join(c.categories[:3])
                line = f"[{i}] {c.title or c.doc_id} — {authors} ({cats}, {c.published or ''})"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def format_inline(citations: list[Citation]) -> dict[int, str]:
        """Return a mapping of citation number → short label for inline use."""
        return {i: (c.title or c.file_name or c.doc_id) for i, c in enumerate(citations, 1)}
