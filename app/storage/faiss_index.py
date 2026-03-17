"""FAISS vector index wrapper."""

from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class FaissIndex:
    """Manages a FAISS flat-IP index (inner-product / cosine after L2-norm)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self._build_empty()

    def _build_empty(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index. Vectors should be L2-normalized for cosine similarity."""
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        logger.info("FAISS index now has %d vectors", self.index.ntotal)

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search the index.

        Args:
            query_vec: (dim,) or (1, dim) float32 vector.
            top_k: Number of results.

        Returns:
            (scores, indices) each of shape (top_k,).
        """
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        return scores[0], indices[0]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        logger.info("FAISS index saved to %s (%d vectors)", path, self.index.ntotal)

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        self.index = faiss.read_index(str(path))
        self.dim = self.index.d
        logger.info("FAISS index loaded from %s (%d vectors)", path, self.index.ntotal)

    def reset(self) -> None:
        self._build_empty()
