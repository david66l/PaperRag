"""Health and config routes."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(request: Request):
    persistence = request.app.state.persistence
    return {
        "status": "ok",
        "index_ready": persistence.is_ready(),
        "documents": persistence.doc_repo.count,
        "chunks": persistence.chunk_repo.count,
        "vectors": persistence.vector_repo.size,
    }


@router.get("/config")
async def config(request: Request):
    settings = request.app.state.settings
    return {
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "fusion_strategy": settings.fusion_strategy,
        "rerank_enabled": settings.rerank_enabled,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "top_k_dense": settings.top_k_dense,
        "top_k_bm25": settings.top_k_bm25,
        "top_n_fused": settings.top_n_fused,
        "top_n_final": settings.top_n_final,
        "top_n_context": settings.top_n_context,
    }


@router.get("/papers/{doc_id}")
async def get_paper(doc_id: str, request: Request):
    persistence = request.app.state.persistence
    doc = persistence.doc_repo.get(doc_id)
    if doc is None:
        return {"error": f"Paper {doc_id} not found"}
    return doc.model_dump()
