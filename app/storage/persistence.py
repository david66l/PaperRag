"""Persistence manager — coordinates saving/loading all repositories."""

from pathlib import Path

from app.core.config import Settings
from app.core.logging import get_logger
from app.storage.repositories.chunk_repository import ChunkRepository
from app.storage.repositories.document_repository import DocumentRepository
from app.storage.repositories.keyword_repository import KeywordIndexRepository
from app.storage.repositories.vector_repository import VectorIndexRepository

logger = get_logger(__name__)


class PersistenceManager:
    """Creates and manages all repository instances."""

    def __init__(self, settings: Settings):
        idx_dir = settings.abs_index_dir
        self.doc_repo = DocumentRepository(idx_dir / "documents.jsonl")
        self.chunk_repo = ChunkRepository(idx_dir / "chunks.jsonl")
        self.vector_repo = VectorIndexRepository(idx_dir / "vector", dim=settings.embedding_dim)
        self.keyword_repo = KeywordIndexRepository(idx_dir / "keyword")

    def save_all(self) -> None:
        logger.info("Saving all indexes...")
        self.doc_repo.save()
        self.chunk_repo.save()
        self.vector_repo.save()
        self.keyword_repo.save()
        logger.info("All indexes saved.")

    def load_all(self) -> None:
        logger.info("Loading all indexes...")
        self.doc_repo.load()
        self.chunk_repo.load()
        self.vector_repo.load()
        self.keyword_repo.load()
        logger.info("All indexes loaded.")

    def is_ready(self) -> bool:
        return self.vector_repo.size > 0 and self.keyword_repo.size > 0
