"""Format citations for display."""

from app.core.schemas import Citation


class CitationFormatter:
    """Format citations into a readable reference list."""

    @staticmethod
    def format_references(citations: list[Citation]) -> str:
        """Generate a numbered reference list."""
        lines = []
        for i, c in enumerate(citations, 1):
            authors = ", ".join(c.authors[:3])
            if len(c.authors) > 3:
                authors += " et al."
            cats = ", ".join(c.categories[:3])
            line = f"[{i}] {c.title} — {authors} ({cats}, {c.update_date})"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def format_inline(citations: list[Citation]) -> dict[int, str]:
        """Return a mapping of citation number → short label for inline use."""
        return {i: c.title for i, c in enumerate(citations, 1)}
