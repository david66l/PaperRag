"""Ingestion pipeline: load → clean → chunk."""

from pathlib import Path

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.schemas import Chunk, PaperDocument
from app.ingestion.chunkers.base import BaseChunker
from app.ingestion.chunkers.metadata_chunker import MetadataPaperChunker
from app.ingestion.loaders.json_loader import JsonLoader
from app.ingestion.preprocess.cleaner import PaperCleaner

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates loading, cleaning and chunking."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.loader = JsonLoader()
        self.cleaner = PaperCleaner()
        self.chunker: BaseChunker = MetadataPaperChunker()

    def run(
        self,
        data_path: str | Path | None = None,
        limit: int = 0,
    ) -> tuple[list[PaperDocument], list[Chunk]]:
        """Execute the full ingestion pipeline.

        Returns:
            (documents, chunks)
        """
        path = Path(data_path) if data_path else self.settings.abs_data_dir / self.settings.default_data_file
        limit = limit or self.settings.load_limit

        # 1. Load
        raw_records = self.loader.load(path, limit=limit)

        # 2. Clean
        documents: list[PaperDocument] = []
        for raw in raw_records:
            doc = self.cleaner.clean(raw)
            if doc:
                documents.append(doc)

        logger.info("Cleaned %d documents", len(documents))

        # 3. Chunk
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunker.chunk(doc))

        logger.info("Generated %d chunks from %d documents", len(all_chunks), len(documents))
        return documents, all_chunks
