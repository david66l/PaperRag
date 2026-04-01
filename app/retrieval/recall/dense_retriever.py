"""Dense (vector) retrieval."""

import numpy as np

from app.core.logging import get_logger
from app.core.schemas import Candidate, SourceScores
from app.embedding.providers.base import BaseEmbeddingProvider
from app.storage.repositories.chunk_repository import ChunkRepository
from app.storage.repositories.vector_repository import VectorIndexRepository

logger = get_logger(__name__)


class DenseRetriever:
    """Retrieves candidates via FAISS vector similarity."""

    def __init__(
        self,
        vector_repo: VectorIndexRepository,
        chunk_repo: ChunkRepository,
        embedding_provider: BaseEmbeddingProvider,
    ):
        self.vector_repo = vector_repo
        self.chunk_repo = chunk_repo
        self.embedding_provider = embedding_provider

    def retrieve(self, query: str, top_k: int) -> list[Candidate]:
        query_vec = self.embedding_provider.embed_query(query)
        results = self.vector_repo.search(query_vec, top_k)

        candidates = []
        for chunk_id, score in results:
            chunk = self.chunk_repo.get(chunk_id)
            if chunk is None:
                continue
            candidates.append(
                Candidate(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    source_type=chunk.source_type,
                    title=chunk.title,
                    paper_id=chunk.paper_id,
                    authors=chunk.authors,
                    categories=chunk.categories,
                    published=chunk.published,
                    file_name=chunk.file_name,
                    file_path=chunk.file_path,
                    page_no=chunk.page_no,
                    source_scores=SourceScores(dense_score=score),
                )
            )
        logger.info("Dense retrieval: %d candidates for query", len(candidates))
        return candidates
