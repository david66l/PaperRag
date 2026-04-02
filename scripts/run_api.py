#!/usr/bin/env python3
"""Script to run the FastAPI server."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from app.core.config import get_settings


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main():
    # OPTIMIZED_BY_CODEX_STEP_1
    settings = get_settings()
    in_docker = _as_bool(os.getenv("PAPERRAG_IN_DOCKER"), default=False)
    host = os.getenv("PAPERRAG_API_HOST") or os.getenv("HOST") or settings.api_host
    port = int(os.getenv("PAPERRAG_API_PORT") or os.getenv("PORT") or settings.api_port)
    reload_flag = _as_bool(
        os.getenv("PAPERRAG_API_RELOAD"),
        default=not in_docker,
    )

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload_flag,
    )


if __name__ == "__main__":
    main()
# STEP_1_SUMMARY: API runtime now supports docker-aware host, port, and reload environment controls.
