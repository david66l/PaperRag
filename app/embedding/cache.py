"""Simple file-based embedding cache to avoid recomputation."""

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """Caches embeddings on disk keyed by a content hash."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def get(self, text: str) -> Optional[np.ndarray]:
        h = self._hash(text)
        path = self.cache_dir / f"{h}.npy"
        if path.exists():
            return np.load(path)
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        h = self._hash(text)
        path = self.cache_dir / f"{h}.npy"
        np.save(path, embedding)

    def get_batch(self, texts: list[str]) -> tuple[list[int], list[np.ndarray], list[int]]:
        """Return (hit_indices, hit_embeddings, miss_indices)."""
        hits_idx, hits_emb, misses = [], [], []
        for i, t in enumerate(texts):
            cached = self.get(t)
            if cached is not None:
                hits_idx.append(i)
                hits_emb.append(cached)
            else:
                misses.append(i)
        return hits_idx, hits_emb, misses

    def put_batch(self, texts: list[str], embeddings: np.ndarray) -> None:
        for t, emb in zip(texts, embeddings):
            self.put(t, emb)
