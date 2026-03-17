"""Ingest / index building routes."""

from fastapi import APIRouter, HTTPException, Request

from app.core.schemas import BuildIndexRequest, BuildIndexResponse

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/build", response_model=BuildIndexResponse)
async def build_index(req: BuildIndexRequest, request: Request) -> BuildIndexResponse:
    """Build or rebuild the knowledge base index."""
    try:
        index_service = request.app.state.index_service
        result = index_service.build(req)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index build failed: {e}")
