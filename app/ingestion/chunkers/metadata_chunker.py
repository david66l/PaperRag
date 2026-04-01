"""Chunker tailored for arXiv metadata into unified chunk schema."""

from app.core.schemas import Chunk, PaperDocument
from app.ingestion.chunkers.base import BaseChunker


class MetadataPaperChunker(BaseChunker):
    """Produces metadata chunks in the unified schema."""

    def chunk(self, doc: PaperDocument) -> list[Chunk]:
        text = f"Title: {doc.title}\nAbstract: {doc.abstract}"
        published = doc.published_date or doc.update_date

        return [
            Chunk(
                chunk_id=f"{doc.doc_id}*meta*0",
                doc_id=doc.doc_id,
                text=text,
                source_type="metadata",
                title=doc.title,
                paper_id=doc.doc_id,
                authors=doc.authors,
                categories=doc.categories,
                published=published,
                file_name=None,
                file_path=None,
                page_no=None,
                metadata={
                    "doi": doc.doi,
                    "journal_ref": doc.journal_ref,
                },
            )
        ]
