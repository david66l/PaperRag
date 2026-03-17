#!/usr/bin/env python3
"""Script to build the index from command line."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.schemas import BuildIndexRequest
from app.services.index_service import IndexService
from app.storage.persistence import PersistenceManager


def main():
    parser = argparse.ArgumentParser(description="Build PaperRAG index")
    parser.add_argument("--data", type=str, help="Path to arXiv JSONL file")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to load (0 = all)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    persistence = PersistenceManager(settings)
    service = IndexService(settings, persistence)

    request = BuildIndexRequest(
        data_path=args.data or "",
        limit=args.limit,
        rebuild=args.rebuild,
    )

    result = service.build(request)
    print(f"\n{result.status}: {result.message}")
    print(f"Documents: {result.num_documents}, Chunks: {result.num_chunks}")
    print(f"Time: {result.elapsed_ms:.0f} ms")


if __name__ == "__main__":
    main()
