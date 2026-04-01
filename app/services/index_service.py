"""Index building service."""

import time

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.schemas import BuildIndexRequest, BuildIndexResponse
from app.embedding.pipeline import EmbeddingPipeline, create_embedding_provider
from app.ingestion.pipeline import IngestionPipeline
from app.storage.persistence import PersistenceManager

logger = get_logger(__name__)


class IndexService:
    """Coordinates the full index build flow."""

    def __init__(self, settings: Settings, persistence: PersistenceManager):
        self.settings = settings
        self.persistence = persistence

    def build(self, request: BuildIndexRequest) -> BuildIndexResponse:
        start = time.time()

        # 1. Full rebuild reset (incremental update intentionally disabled)
        self.persistence.doc_repo.clear()
        self.persistence.chunk_repo.clear()
        self.persistence.vector_repo.reset()
        self.persistence.keyword_repo.reset()

        # 2. Ingestion
        ingestion = IngestionPipeline(self.settings)
        data_path = request.data_path or None
        limit = request.limit or 0
        documents, chunks = ingestion.run(data_path=data_path, limit=limit)

        if not chunks:
            return BuildIndexResponse(
                status="error",
                message="No valid chunks produced from ingestion",
                elapsed_ms=(time.time() - start) * 1000,
            )

        # 3. Embedding
        provider = create_embedding_provider(self.settings)
        emb_pipeline = EmbeddingPipeline(provider, batch_size=self.settings.embedding_batch_size)
        chunk_ids, vectors = emb_pipeline.embed_chunks(chunks)

        # 4. Store documents + chunks (chunk_id -> chunk metadata mapping in chunk_repo)
        self.persistence.doc_repo.add_batch(documents)
        self.persistence.chunk_repo.add_batch(chunks)

        # 5. Build vector index
        self.persistence.vector_repo.build(chunk_ids, vectors)

        # 6. Build keyword index
        texts = [c.text for c in chunks]
        self.persistence.keyword_repo.build(chunk_ids, texts)

        # 7. Persist
        self.persistence.save_all()

        elapsed = (time.time() - start) * 1000
        logger.info("Index built: %d docs, %d chunks in %.1f ms", len(documents), len(chunks), elapsed)

        return BuildIndexResponse(
            status="success",
            num_documents=len(documents),
            num_chunks=len(chunks),
            elapsed_ms=elapsed,
            message=f"Successfully indexed {len(documents)} documents ({len(chunks)} chunks)",
        )
