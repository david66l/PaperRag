"""PDF loader for local full-text ingestion."""

from __future__ import annotations

import re
from pathlib import Path

from app.core.logging import get_logger
from app.core.schemas import Chunk, PaperDocument

logger = get_logger(__name__)


class PDFLoader:
    """Recursively scan local PDFs and convert them into unified chunks."""

    def __init__(
        self,
        pdf_dir: Path,
        max_files: int = 100,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.max_files = max(0, max_files)
        self.chunk_size = max(100, chunk_size)
        self.chunk_overlap = max(0, chunk_overlap)

    def load(self) -> tuple[list[PaperDocument], list[Chunk]]:
        if not self.pdf_dir.exists():
            logger.warning("PDF dir does not exist, skipping PDF ingestion: %s", self.pdf_dir)
            return [], []

        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF (fitz) not installed, skipping PDF ingestion")
            return [], []

        files = self._scan_pdf_files()
        if self.max_files:
            files = files[: self.max_files]

        logger.info("PDF ingestion scanning %d files from %s", len(files), self.pdf_dir)

        documents: list[PaperDocument] = []
        chunks: list[Chunk] = []

        for pdf_path in files:
            doc_id = pdf_path.stem
            file_name = pdf_path.name
            file_path = str(pdf_path)

            try:
                pdf_doc = fitz.open(pdf_path)
            except Exception as exc:
                logger.warning("Skip PDF parse failure: %s (%s)", pdf_path, exc)
                continue

            chunk_count = 0
            try:
                for page_idx in range(pdf_doc.page_count):
                    page_no = page_idx + 1
                    try:
                        page = pdf_doc.load_page(page_idx)
                        raw_text = page.get_text("text")
                    except Exception as exc:
                        logger.warning("Skip page parse failure: %s page=%d (%s)", pdf_path, page_no, exc)
                        continue

                    text = self._clean_text(raw_text)
                    if not text:
                        continue

                    page_chunks = self._split_text(text)
                    for chunk_idx, piece in enumerate(page_chunks):
                        chunks.append(
                            Chunk(
                                chunk_id=f"{doc_id}_p{page_no}_c{chunk_idx}",
                                doc_id=doc_id,
                                text=piece,
                                source_type="pdf",
                                title=doc_id,
                                paper_id=None,
                                categories=[],
                                authors=[],
                                published=None,
                                file_name=file_name,
                                file_path=file_path,
                                page_no=page_no,
                                metadata={},
                            )
                        )
                        chunk_count += 1
            finally:
                pdf_doc.close()

            if chunk_count == 0:
                continue

            documents.append(
                PaperDocument(
                    doc_id=doc_id,
                    title=doc_id,
                    abstract="",
                    content=f"PDF: {file_name}",
                    authors=[],
                    categories=[],
                    update_date="",
                    published_date="",
                    doi="",
                    journal_ref="",
                    metadata={
                        "source_type": "pdf",
                        "file_name": file_name,
                        "file_path": file_path,
                    },
                )
            )

        logger.info("PDF ingestion complete: %d docs, %d chunks", len(documents), len(chunks))
        return documents, chunks

    def _scan_pdf_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self.pdf_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".pdf":
                files.append(path)
        files.sort(key=lambda p: str(p))
        return files

    @staticmethod
    def _clean_text(text: str | None) -> str:
        if not text:
            return ""
        safe = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        safe = re.sub(r"\s+", " ", safe).strip()
        return safe

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        overlap = min(self.chunk_overlap, self.chunk_size - 1)
        step = max(1, self.chunk_size - overlap)
        out: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            piece = text[start:end].strip()
            if piece:
                out.append(piece)
            if end >= len(text):
                break
            start += step
        return out
