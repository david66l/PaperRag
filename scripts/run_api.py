#!/usr/bin/env python3
"""Script to run the FastAPI server."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from app.core.config import get_settings


def main():
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    main()
