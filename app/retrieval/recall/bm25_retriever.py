"""BM25 keyword retrieval."""

from app.core.logging import get_logger
from app.core.schemas import Candidate, SourceScores
from app.storage.repositories.chunk_repository import ChunkRepository
from app.storage.repositories.keyword_repository import KeywordIndexRepository

logger = get_logger(__name__)


class BM25Retriever:
    """Retrieves candidates via BM25 keyword matching."""

    def __init__(
        self,
        keyword_repo: KeywordIndexRepository,
        chunk_repo: ChunkRepository,
    ):
        self.keyword_repo = keyword_repo
        self.chunk_repo = chunk_repo

    def retrieve(self, query: str, top_k: int) -> list[Candidate]:
        results = self.keyword_repo.search(query, top_k)

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
                    source_scores=SourceScores(bm25_score=score),
                )
            )
        logger.info("BM25 retrieval: %d candidates for query", len(candidates))
        return candidates
