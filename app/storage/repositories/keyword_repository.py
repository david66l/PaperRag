"""Keyword (BM25) index repository."""

from pathlib import Path

from app.core.logging import get_logger
from app.storage.bm25_index import BM25Index

logger = get_logger(__name__)


class KeywordIndexRepository:
    """Wraps the BM25 index for persistence and retrieval."""

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.bm25_idx = BM25Index()

    @property
    def size(self) -> int:
        return self.bm25_idx.size

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        self.bm25_idx.build(chunk_ids, texts)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return self.bm25_idx.search(query, top_k)

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_idx.save(self.index_dir / "bm25.json")

    def load(self) -> None:
        self.bm25_idx.load(self.index_dir / "bm25.json")

    def reset(self) -> None:
        self.bm25_idx.reset()
