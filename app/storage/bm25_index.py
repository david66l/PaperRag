"""BM25 keyword index wrapper."""

import json
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from app.core.logging import get_logger

logger = get_logger(__name__)


def simple_tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer with basic cleanup."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25Index:
    """Wrapper around rank_bm25 for keyword retrieval."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: list[str] = []
        self.corpus: list[list[str]] = []

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Build BM25 index from texts."""
        self.chunk_ids = chunk_ids
        self.corpus = [simple_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.corpus)
        logger.info("BM25 index built with %d documents", len(self.chunk_ids))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) sorted by relevance."""
        if self.bm25 is None:
            return []
        tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunk_ids[idx], float(scores[idx])))
        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunk_ids": self.chunk_ids,
            "corpus": self.corpus,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("BM25 index saved to %s", path)

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunk_ids = data["chunk_ids"]
        self.corpus = data["corpus"]
        self.bm25 = BM25Okapi(self.corpus)
        logger.info("BM25 index loaded from %s (%d docs)", path, len(self.chunk_ids))

    def reset(self) -> None:
        self.bm25 = None
        self.chunk_ids = []
        self.corpus = []
