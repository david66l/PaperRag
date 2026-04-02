#!/usr/bin/env python3
"""Script to build the index from command line."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.schemas import BuildIndexRequest
from app.services.index_service import IndexService
from app.storage.index_stats import IndexStatsService
from app.storage.persistence import PersistenceManager


def main():
    # OPTIMIZED_BY_CODEX_STEP_3
    parser = argparse.ArgumentParser(description="Build PaperRAG index")
    parser.add_argument("--data", type=str, help="Path to arXiv JSONL file")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to load (0 = all)")
    parser.add_argument("--max_papers", type=int, default=20000, help="Default max papers when --limit is not set")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    persistence = PersistenceManager(settings)
    service = IndexService(settings, persistence)
    stats_service = IndexStatsService(settings, persistence)

    effective_limit = args.limit if args.limit > 0 else args.max_papers
    started_at = datetime.now()

    request = BuildIndexRequest(
        data_path=args.data or "",
        limit=effective_limit,
        rebuild=args.rebuild,
    )

    result = service.build(request)
    stats = stats_service.snapshot()
    bench = stats_service.benchmark_retrieval_p95(top_k=min(10, max(1, settings.top_k_dense)), sample_size=20)

    print(f"\n{result.status}: {result.message}")
    print(f"Documents: {result.num_documents}, Chunks: {result.num_chunks}")
    print(f"Max papers (effective): {effective_limit}")
    print(f"Time: {result.elapsed_ms:.0f} ms")
    print(f"Index size: {stats['index_size_mb']:.2f} MB ({int(stats['index_size_bytes'])} bytes)")
    print(f"Memory usage: {stats['memory_mb']:.2f} MB")
    print(f"p95 retrieval latency: {bench['p95_retrieval_ms']:.2f} ms (avg: {bench['avg_retrieval_ms']:.2f} ms)")
    print(f"Started at: {started_at.isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
# STEP_3_SUMMARY: Build script now supports max_papers and prints index size, memory, runtime, and retrieval latency metrics.
