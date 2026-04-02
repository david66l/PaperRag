"""Index statistics and retrieval benchmark helpers."""

# OPTIMIZED_BY_CODEX_STEP_3
from __future__ import annotations

import math
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.embedding.pipeline import create_embedding_provider
from app.retrieval.pipeline import RetrievalPipeline
from app.storage.persistence import PersistenceManager

logger = get_logger(__name__)


class IndexStatsService:
    """Provide index-level metrics for scripts and operational checks."""

    def __init__(self, settings: Settings, persistence: PersistenceManager):
        self.settings = settings
        self.persistence = persistence

    def snapshot(self) -> dict[str, Any]:
        index_size_bytes = self._directory_size(self.settings.abs_index_dir)
        return {
            "documents": self.persistence.doc_repo.count,
            "chunks": self.persistence.chunk_repo.count,
            "vectors": self.persistence.vector_repo.size,
            "index_size_bytes": index_size_bytes,
            "index_size_mb": index_size_bytes / (1024 * 1024),
            "memory_mb": self._memory_usage_mb(),
        }

    def benchmark_retrieval_p95(
        self,
        top_k: int = 5,
        sample_size: int = 20,
    ) -> dict[str, float]:
        if not self.persistence.is_ready() or self.persistence.chunk_repo.count == 0:
            return {
                "p95_retrieval_ms": 0.0,
                "avg_retrieval_ms": 0.0,
                "samples": 0.0,
            }

        queries = self._make_sample_queries(sample_size)
        if not queries:
            return {
                "p95_retrieval_ms": 0.0,
                "avg_retrieval_ms": 0.0,
                "samples": 0.0,
            }

        provider = create_embedding_provider(self.settings)
        pipeline = RetrievalPipeline(self.settings, self.persistence, provider)

        latencies_ms: list[float] = []
        for query in queries:
            start = time.perf_counter()
            pipeline.run(query, top_k=top_k)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

        latencies_ms.sort()
        p95_idx = max(0, min(len(latencies_ms) - 1, math.ceil(0.95 * len(latencies_ms)) - 1))
        p95 = latencies_ms[p95_idx]

        return {
            "p95_retrieval_ms": float(p95),
            "avg_retrieval_ms": float(mean(latencies_ms)),
            "samples": float(len(latencies_ms)),
        }

    def _make_sample_queries(self, sample_size: int) -> list[str]:
        chunks = self.persistence.chunk_repo.get_all()
        if not chunks:
            return []

        queries: list[str] = []
        for chunk in chunks[: max(1, sample_size)]:
            title = chunk.title or chunk.doc_id
            words = re.findall(r"[A-Za-z0-9\-]{3,}", chunk.text)
            phrase = " ".join(words[:8]) if words else "core idea"
            queries.append(f"Summarize {title} about {phrase}")
        return queries[:sample_size]

    @staticmethod
    def _directory_size(path: Path) -> int:
        total = 0
        if not path.exists():
            return 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                try:
                    total += file_path.stat().st_size
                except OSError:
                    continue
        return total

    @staticmethod
    def _memory_usage_mb() -> float:
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return float(usage) / (1024 * 1024)
            return float(usage) / 1024
        except Exception as exc:
            logger.debug("Memory usage metric unavailable: %s", exc)
            return 0.0


# STEP_3_SUMMARY: Added index metrics API with snapshot and p95 retrieval benchmark utilities.
