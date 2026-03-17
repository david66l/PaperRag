"""Chunk-level storage."""

import json
from pathlib import Path
from typing import Optional

from app.core.logging import get_logger
from app.core.schemas import Chunk

logger = get_logger(__name__)


class ChunkRepository:
    """Persists Chunk objects as JSONL. Provides chunk_id → Chunk lookup."""

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self._chunks: dict[str, Chunk] = {}

    @property
    def count(self) -> int:
        return len(self._chunks)

    def add_batch(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.chunk_id] = c

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)

    def get_all(self) -> list[Chunk]:
        return list(self._chunks.values())

    def get_ids(self) -> list[str]:
        return list(self._chunks.keys())

    def get_texts(self) -> list[str]:
        return [c.text for c in self._chunks.values()]

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            for chunk in self._chunks.values():
                f.write(chunk.model_dump_json() + "\n")
        logger.info("Saved %d chunks to %s", len(self._chunks), self.storage_path)

    def load(self) -> None:
        if not self.storage_path.exists():
            logger.warning("Chunk store not found: %s", self.storage_path)
            return
        self._chunks.clear()
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunk = Chunk.model_validate_json(line)
                    self._chunks[chunk.chunk_id] = chunk
        logger.info("Loaded %d chunks from %s", len(self._chunks), self.storage_path)

    def clear(self) -> None:
        self._chunks.clear()
