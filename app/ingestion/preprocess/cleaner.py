"""Data cleaning and normalization for raw paper records."""

import re
from typing import Optional

from app.core.logging import get_logger
from app.core.schemas import PaperDocument, RawPaperRecord

logger = get_logger(__name__)


class PaperCleaner:
    """Convert RawPaperRecord → PaperDocument with cleaning."""

    # Minimum abstract length to be considered valid
    MIN_ABSTRACT_LEN = 30

    def clean(self, raw: RawPaperRecord) -> Optional[PaperDocument]:
        """Return a PaperDocument or None if the record is invalid."""
        # --- basic validity ---
        if not raw.id or not raw.title:
            logger.debug("Skip record: missing id or title")
            return None

        abstract = self._clean_text(raw.abstract)
        if len(abstract) < self.MIN_ABSTRACT_LEN:
            logger.debug("Skip %s: abstract too short (%d chars)", raw.id, len(abstract))
            return None

        title = self._clean_text(raw.title)
        authors = self._parse_authors(raw)
        categories = self._parse_categories(raw.categories)
        update_date = self._normalize_date(raw.update_date)
        published_date = self._extract_published_date(raw)

        content = f"{title}\n\n{abstract}"

        return PaperDocument(
            doc_id=raw.id,
            title=title,
            abstract=abstract,
            content=content,
            authors=authors,
            categories=categories,
            update_date=update_date,
            published_date=published_date,
            doi=raw.doi.strip(),
            journal_ref=raw.journal_ref.strip(),
            metadata={
                "submitter": raw.submitter,
                "comments": raw.comments,
                "license": raw.license,
            },
        )

    # ── helpers ──────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _parse_authors(raw: RawPaperRecord) -> list[str]:
        if raw.authors_parsed:
            names = []
            for parts in raw.authors_parsed:
                # parts = [last, first, suffix]
                name_parts = [p.strip() for p in parts if p.strip()]
                if name_parts:
                    names.append(" ".join(reversed(name_parts)))
            return names
        # Fallback: split the flat authors string
        if raw.authors:
            return [a.strip() for a in re.split(r",\s*and\s*|,\s*|\s+and\s+", raw.authors) if a.strip()]
        return []

    @staticmethod
    def _parse_categories(cats: str) -> list[str]:
        if not cats:
            return []
        return [c.strip() for c in cats.split() if c.strip()]

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        date_str = date_str.strip()
        # Already ISO-like
        if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            return date_str[:10]
        return date_str

    @staticmethod
    def _extract_published_date(raw: RawPaperRecord) -> str:
        if raw.versions:
            # First version is the original submission
            v1 = raw.versions[0]
            created = v1.get("created", "")
            if created:
                return created
        return ""
