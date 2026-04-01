"""Core data models used across the entire system."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ────────────────────────────────────────────
# Ingestion models
# ────────────────────────────────────────────

class RawPaperRecord(BaseModel):
    """Raw record directly from arXiv JSON."""
    id: str = ""
    submitter: str = ""
    authors: str = ""
    title: str = ""
    comments: str = ""
    journal_ref: str = ""  # journal-ref
    doi: str = ""
    report_no: str = ""  # report-no
    categories: str = ""
    license: str = ""
    abstract: str = ""
    versions: list[dict[str, Any]] = Field(default_factory=list)
    update_date: str = ""
    authors_parsed: list[list[str]] = Field(default_factory=list)


class PaperDocument(BaseModel):
    """Cleaned, standardized paper document."""
    doc_id: str
    title: str
    abstract: str
    content: str  # title + abstract combined
    authors: list[str]
    categories: list[str]
    update_date: str  # ISO date string
    published_date: str = ""
    doi: str = ""
    journal_ref: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────
# Chunk models
# ────────────────────────────────────────────

class Chunk(BaseModel):
    """A chunk ready for embedding."""
    chunk_id: str
    doc_id: str
    text: str
    source_type: Literal["metadata", "pdf"] = "metadata"
    title: str | None = None
    paper_id: str | None = None
    categories: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    published: str | None = None
    file_name: str | None = None
    file_path: str | None = None
    page_no: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────
# Retrieval models
# ────────────────────────────────────────────

class SourceScores(BaseModel):
    dense_score: float = 0.0
    bm25_score: float = 0.0
    metadata_score: float = 0.0


class Candidate(BaseModel):
    """A retrieval candidate from one or more recall paths."""
    chunk_id: str
    doc_id: str
    text: str
    source_type: Literal["metadata", "pdf"] = "metadata"
    score: float = 0.0
    title: str | None = None
    paper_id: str | None = None
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published: str | None = None
    file_name: str | None = None
    file_path: str | None = None
    page_no: int | None = None
    source_scores: SourceScores = Field(default_factory=SourceScores)
    fused_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class Citation(BaseModel):
    """A citation reference in the final answer."""
    chunk_id: str
    doc_id: str
    source_type: Literal["metadata", "pdf"] = "metadata"
    title: str | None = None
    paper_id: str | None = None
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published: str | None = None
    file_name: str | None = None
    file_path: str | None = None
    page_no: int | None = None
    relevance_score: float = 0.0


class RetrievalResult(BaseModel):
    """Full retrieval result passed to the generation module."""
    query: str
    candidates: list[Candidate]
    context_text: str
    citations: list[Citation]
    trace: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────
# API models
# ────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: Literal["concise", "analysis"] = "concise"


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[Candidate]
    evidence_level: Literal["metadata", "pdf", "hybrid"] = "metadata"
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: float = 0.0


class BuildIndexRequest(BaseModel):
    data_path: str = ""
    limit: int = 0
    rebuild: bool = False


class BuildIndexResponse(BaseModel):
    status: str
    num_documents: int = 0
    num_chunks: int = 0
    elapsed_ms: float = 0.0
    message: str = ""
