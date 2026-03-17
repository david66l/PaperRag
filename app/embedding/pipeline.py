"""Embedding pipeline: embed chunks using configured provider."""

import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.schemas import Chunk
from app.embedding.providers.base import BaseEmbeddingProvider
from app.embedding.providers.local_provider import LocalEmbeddingProvider
from app.embedding.providers.api_provider import APIEmbeddingProvider

logger = get_logger(__name__)


def create_embedding_provider(settings: Settings) -> BaseEmbeddingProvider:
    """Factory function to create the right embedding provider."""
    if settings.embedding_provider == "api":
        return APIEmbeddingProvider(
            api_url=settings.embedding_api_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dim,
        )
    return LocalEmbeddingProvider(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dim,
    )


class EmbeddingPipeline:
    """Embed a list of chunks and return (chunk_ids, vectors)."""

    def __init__(self, provider: BaseEmbeddingProvider, batch_size: int = 256):
        self.provider = provider
        self.batch_size = batch_size

    def embed_chunks(self, chunks: list[Chunk]) -> tuple[list[str], np.ndarray]:
        """Embed all chunks.

        Returns:
            (chunk_ids, embeddings) where embeddings is (N, dim) float32.
        """
        texts = [c.text for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]

        all_embs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            emb = self.provider.embed_documents(batch)
            all_embs.append(emb)
            logger.info("Embedded batch %d/%d", i // self.batch_size + 1,
                        (len(texts) + self.batch_size - 1) // self.batch_size)

        embeddings = np.concatenate(all_embs, axis=0) if all_embs else np.empty((0, self.provider.dim), dtype=np.float32)

        assert embeddings.shape == (len(chunks), self.provider.dim), (
            f"Shape mismatch: {embeddings.shape} vs ({len(chunks)}, {self.provider.dim})"
        )
        return chunk_ids, embeddings
