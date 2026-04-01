"""JSON / JSONL loader for arXiv metadata."""

import json
from pathlib import Path
from typing import Iterator

from app.core.logging import get_logger
from app.core.schemas import RawPaperRecord
from app.ingestion.loaders.base import BaseLoader

logger = get_logger(__name__)


class JsonLoader(BaseLoader):
    """Load arXiv metadata from a JSON-lines file (one JSON object per line)
    or a standard JSON array file.
    """

    def load(self, path: Path, limit: int = 0) -> Iterator[RawPaperRecord]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        logger.info("Loading data from %s (limit=%s)", path, limit or "unlimited")
        count = 0

        with open(path, "r", encoding="utf-8") as fh:
            # Try JSONL first (most common for arXiv dump)
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    # arXiv dump uses "journal-ref" and "report-no" with hyphens
                    if "journal-ref" in raw:
                        raw["journal_ref"] = raw.pop("journal-ref")
                    if "report-no" in raw:
                        raw["report_no"] = raw.pop("report-no")
                    # Normalize nullable string fields from arXiv dump.
                    for key in (
                        "submitter",
                        "authors",
                        "title",
                        "comments",
                        "journal_ref",
                        "doi",
                        "report_no",
                        "categories",
                        "license",
                        "abstract",
                        "update_date",
                    ):
                        if raw.get(key) is None:
                            raw[key] = ""
                    record = RawPaperRecord.model_validate(raw)
                    yield record
                    count += 1
                    if limit and count >= limit:
                        break
                except Exception as exc:
                    logger.warning("Skip line %d: %s", line_no, exc)
                    continue

        logger.info("Loaded %d records from %s", count, path.name)
