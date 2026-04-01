"""Ingestion pipeline: load → clean → chunk."""

from pathlib import Path

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.schemas import Chunk, PaperDocument
from app.ingestion.chunkers.base import BaseChunker
from app.ingestion.chunkers.metadata_chunker import MetadataPaperChunker
from app.ingestion.loaders.json_loader import JsonLoader
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.preprocess.cleaner import PaperCleaner

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates loading, cleaning and chunking."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.loader = JsonLoader()
        self.cleaner = PaperCleaner()
        self.chunker: BaseChunker = MetadataPaperChunker()
        self.pdf_loader = PDFLoader(
            pdf_dir=settings.abs_pdf_dir,
            max_files=settings.pdf_max_files,
            chunk_size=settings.pdf_chunk_size,
            chunk_overlap=settings.pdf_chunk_overlap,
        )

    def run(
        self,
        data_path: str | Path | None = None,
        limit: int = 0,
    ) -> tuple[list[PaperDocument], list[Chunk]]:
        """Execute the full ingestion pipeline.

        Returns:
            (documents, chunks)
        """
        metadata_docs, metadata_chunks = self._ingest_metadata(data_path=data_path, limit=limit)
        pdf_docs, pdf_chunks = self._ingest_pdf()

        all_docs = metadata_docs + pdf_docs
        all_chunks = metadata_chunks + pdf_chunks
        logger.info(
            "Unified ingestion complete: metadata=%d chunks, pdf=%d chunks, total=%d chunks",
            len(metadata_chunks),
            len(pdf_chunks),
            len(all_chunks),
        )
        return all_docs, all_chunks

    def _ingest_metadata(
        self,
        data_path: str | Path | None = None,
        limit: int = 0,
    ) -> tuple[list[PaperDocument], list[Chunk]]:
        path = Path(data_path) if data_path else self.settings.abs_data_dir / self.settings.default_data_file
        limit = limit or self.settings.load_limit
        if not path.exists():
            logger.warning("Metadata file not found, skipping metadata ingestion: %s", path)
            return [], []

        raw_records = self.loader.load(path, limit=limit)

        documents: list[PaperDocument] = []
        for raw in raw_records:
            doc = self.cleaner.clean(raw)
            if doc:
                documents.append(doc)

        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.chunker.chunk(doc))

        logger.info("Metadata ingestion complete: %d docs, %d chunks", len(documents), len(chunks))
        return documents, chunks

    def _ingest_pdf(self) -> tuple[list[PaperDocument], list[Chunk]]:
        if not self.settings.enable_pdf_ingest:
            logger.info("PDF ingestion disabled by PAPERRAG_ENABLE_PDF_INGEST")
            return [], []
        return self.pdf_loader.load()
