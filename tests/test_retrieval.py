"""Tests for recall retrievers with mocked repositories."""

# OPTIMIZED_BY_CODEX_STEP_4
import numpy as np

from app.core.schemas import Chunk
from app.retrieval.recall.bm25_retriever import BM25Retriever
from app.retrieval.recall.dense_retriever import DenseRetriever


class DummyChunkRepo:
    def __init__(self, chunks: dict[str, Chunk]):
        self._chunks = chunks

    def get(self, chunk_id: str):
        return self._chunks.get(chunk_id)


class DummyKeywordRepo:
    def search(self, query: str, top_k: int):
        return [("m1", 2.0), ("missing", 1.0)][:top_k]


class DummyVectorRepo:
    def search(self, query_vec, top_k: int):
        assert isinstance(query_vec, np.ndarray)
        return [("d1", 0.8), ("missing", 0.2)][:top_k]


class DummyEmbedProvider:
    def embed_query(self, query: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


def test_bm25_retriever_returns_existing_chunks_only():
    chunk_repo = DummyChunkRepo(
        {
            "m1": Chunk(chunk_id="m1", doc_id="doc-m", text="meta text", source_type="metadata", title="T"),
        }
    )
    retriever = BM25Retriever(keyword_repo=DummyKeywordRepo(), chunk_repo=chunk_repo)

    candidates = retriever.retrieve("query", top_k=5)

    assert len(candidates) == 1
    assert candidates[0].chunk_id == "m1"
    assert candidates[0].source_scores.bm25_score == 2.0


def test_dense_retriever_returns_existing_chunks_only():
    chunk_repo = DummyChunkRepo(
        {
            "d1": Chunk(chunk_id="d1", doc_id="doc-d", text="dense text", source_type="pdf", file_name="x.pdf"),
        }
    )
    retriever = DenseRetriever(
        vector_repo=DummyVectorRepo(),
        chunk_repo=chunk_repo,
        embedding_provider=DummyEmbedProvider(),
    )

    candidates = retriever.retrieve("query", top_k=5)

    assert len(candidates) == 1
    assert candidates[0].chunk_id == "d1"
    assert candidates[0].source_type == "pdf"
    assert candidates[0].source_scores.dense_score == 0.8


# STEP_4_SUMMARY: Added retriever tests ensuring safe handling of missing chunk mappings.
