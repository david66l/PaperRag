"""Tests for embedding pipeline behavior."""

# OPTIMIZED_BY_CODEX_STEP_4
import numpy as np

from app.core.schemas import Chunk
from app.embedding.pipeline import EmbeddingPipeline
from app.embedding.providers.base import BaseEmbeddingProvider


class DummyProvider(BaseEmbeddingProvider):
    @property
    def dim(self) -> int:
        return 4

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        arr = []
        for text in texts:
            arr.append([float(len(text)), 1.0, 0.0, 0.5])
        return np.array(arr, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return np.array([float(len(query)), 1.0, 0.0, 0.5], dtype=np.float32)


def test_embedding_pipeline_returns_expected_shape_and_ids():
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="hello", source_type="metadata"),
        Chunk(chunk_id="c2", doc_id="d2", text="world!", source_type="pdf"),
    ]
    pipeline = EmbeddingPipeline(provider=DummyProvider(), batch_size=1)

    chunk_ids, vectors = pipeline.embed_chunks(chunks)

    assert chunk_ids == ["c1", "c2"]
    assert vectors.shape == (2, 4)
    assert vectors.dtype == np.float32
    assert vectors[0, 0] == 5.0
    assert vectors[1, 0] == 6.0


# STEP_4_SUMMARY: Added deterministic embedding pipeline tests with a mock provider.
