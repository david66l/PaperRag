"""Chunker tailored for arXiv metadata (title + abstract)."""

from app.core.schemas import Chunk, PaperDocument
from app.ingestion.chunkers.base import BaseChunker


class MetadataPaperChunker(BaseChunker):
    """Produces 1-2 chunks per paper from metadata fields.

    Chunk 1 (always): title + abstract combined.
    Chunk 2 (optional): abstract only, enabled by *include_abstract_only*.
    """

    def __init__(self, include_abstract_only: bool = False):
        self.include_abstract_only = include_abstract_only

    def chunk(self, doc: PaperDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        base_meta = {
            "doi": doc.doi,
            "journal_ref": doc.journal_ref,
            "published_date": doc.published_date,
        }

        # Primary chunk: title + abstract
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::0",
                doc_id=doc.doc_id,
                text=doc.content,  # title \n\n abstract
                title=doc.title,
                authors=doc.authors,
                categories=doc.categories,
                update_date=doc.update_date,
                chunk_type="title_abstract",
                metadata=base_meta,
            )
        )

        if self.include_abstract_only and len(doc.abstract) > 100:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::1",
                    doc_id=doc.doc_id,
                    text=doc.abstract,
                    title=doc.title,
                    authors=doc.authors,
                    categories=doc.categories,
                    update_date=doc.update_date,
                    chunk_type="abstract_only",
                    metadata=base_meta,
                )
            )

        return chunks
