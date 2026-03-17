"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.query import router as query_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.index_service import IndexService
from app.services.query_service import QueryService
from app.storage.persistence import PersistenceManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    setup_logging()

    persistence = PersistenceManager(settings)

    # Try to load existing indexes
    try:
        persistence.load_all()
    except Exception:
        pass  # Indexes not built yet — that's fine

    app.state.settings = settings
    app.state.persistence = persistence
    app.state.index_service = IndexService(settings, persistence)
    app.state.query_service = QueryService(settings, persistence)

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="PaperRAG — AI Paper Knowledge Base",
        description="RAG system for querying arXiv paper metadata",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    return app


app = create_app()
