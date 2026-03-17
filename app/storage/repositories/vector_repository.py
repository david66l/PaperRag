"""Vector index repository — wraps FAISS index + chunk_id mapping."""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.logging import get_logger
from app.storage.faiss_index import FaissIndex

logger = get_logger(__name__)


class VectorIndexRepository:
    """Manages FAISS index + the ordered chunk_id list that maps
    internal FAISS row positions to chunk_ids.
    """

    def __init__(self, index_dir: Path, dim: int):
        self.index_dir = Path(index_dir)
        self.faiss_idx = FaissIndex(dim=dim)
        self.chunk_ids: list[str] = []

    @property
    def size(self) -> int:
        return self.faiss_idx.size

    def build(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        self.faiss_idx.reset()
        self.chunk_ids = chunk_ids
        self.faiss_idx.add(vectors)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score)."""
        scores, indices = self.faiss_idx.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            results.append((self.chunk_ids[idx], float(score)))
        return results

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_idx.save(self.index_dir / "faiss.index")
        with open(self.index_dir / "chunk_ids.json", "w") as f:
            json.dump(self.chunk_ids, f)

    def load(self) -> None:
        self.faiss_idx.load(self.index_dir / "faiss.index")
        with open(self.index_dir / "chunk_ids.json", "r") as f:
            self.chunk_ids = json.load(f)
        logger.info("VectorIndexRepository loaded: %d vectors", self.size)

    def reset(self) -> None:
        self.faiss_idx.reset()
        self.chunk_ids = []
