"""Query routes."""

from fastapi import APIRouter, HTTPException, Request

from app.core.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request) -> QueryResponse:
    """Ask a question against the paper knowledge base."""
    persistence = request.app.state.persistence
    if not persistence.is_ready():
        raise HTTPException(status_code=503, detail="Index not built yet. Call POST /ingest/build first.")

    try:
        query_service = request.app.state.query_service
        result = query_service.query(query=req.query, top_k=req.top_k, mode=req.mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
