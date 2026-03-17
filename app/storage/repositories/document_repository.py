"""Document-level storage."""

import json
from pathlib import Path
from typing import Optional

from app.core.logging import get_logger
from app.core.schemas import PaperDocument

logger = get_logger(__name__)


class DocumentRepository:
    """Persists PaperDocument objects as JSONL."""

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self._docs: dict[str, PaperDocument] = {}

    @property
    def count(self) -> int:
        return len(self._docs)

    def add(self, doc: PaperDocument) -> None:
        self._docs[doc.doc_id] = doc

    def add_batch(self, docs: list[PaperDocument]) -> None:
        for d in docs:
            self._docs[d.doc_id] = d

    def get(self, doc_id: str) -> Optional[PaperDocument]:
        return self._docs.get(doc_id)

    def get_all(self) -> list[PaperDocument]:
        return list(self._docs.values())

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            for doc in self._docs.values():
                f.write(doc.model_dump_json() + "\n")
        logger.info("Saved %d documents to %s", len(self._docs), self.storage_path)

    def load(self) -> None:
        if not self.storage_path.exists():
            logger.warning("Document store not found: %s", self.storage_path)
            return
        self._docs.clear()
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = PaperDocument.model_validate_json(line)
                    self._docs[doc.doc_id] = doc
        logger.info("Loaded %d documents from %s", len(self._docs), self.storage_path)

    def clear(self) -> None:
        self._docs.clear()
