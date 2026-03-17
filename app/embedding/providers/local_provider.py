"""Local embedding provider using sentence-transformers."""

import numpy as np

from app.core.logging import get_logger

from .base import BaseEmbeddingProvider

logger = get_logger(__name__)


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Uses sentence-transformers models loaded locally."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dimension: int = 384):
        self._model_name = model_name
        self._dim = dimension
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            # Update dim from model
            self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        logger.info("Embedding %d documents", len(texts))
        embeddings = self._model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            batch_size=64,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        self._load_model()
        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(embedding[0], dtype=np.float32)
