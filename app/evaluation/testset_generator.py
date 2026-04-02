"""Test query generation for ablation evaluation."""

# OPTIMIZED_BY_CODEX_STEP_2
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from app.core.schemas import Chunk


@dataclass
class EvalCase:
    """One evaluation case used by the ablation runner."""

    query: str
    ground_truth: str
    target_chunk_ids: set[str]
    expected_source: str


class TestsetGenerator:
    """Generate synthetic evaluation queries from indexed chunks."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, chunks: list[Chunk], num_queries: int = 50) -> list[EvalCase]:
        if not chunks:
            return []

        metadata_chunks = [c for c in chunks if c.source_type == "metadata"]
        pdf_chunks = [c for c in chunks if c.source_type == "pdf"]

        cases: list[EvalCase] = []
        half = max(1, num_queries // 2)

        cases.extend(self._sample_cases(metadata_chunks or chunks, half, preferred_source="metadata"))
        cases.extend(self._sample_cases(pdf_chunks or chunks, num_queries - len(cases), preferred_source="pdf"))

        while len(cases) < num_queries:
            c = self._rng.choice(chunks)
            cases.append(self._build_case(c, preferred_source=c.source_type))

        self._rng.shuffle(cases)
        return cases[:num_queries]

    def _sample_cases(self, source_chunks: list[Chunk], n: int, preferred_source: str) -> list[EvalCase]:
        if not source_chunks:
            return []

        if len(source_chunks) <= n:
            picked = [self._rng.choice(source_chunks) for _ in range(n)]
        else:
            picked = self._rng.sample(source_chunks, n)

        return [self._build_case(chunk, preferred_source=preferred_source) for chunk in picked]

    def _build_case(self, chunk: Chunk, preferred_source: str) -> EvalCase:
        keywords = self._extract_keywords(chunk.text)
        k1 = keywords[0] if keywords else "method"
        k2 = keywords[1] if len(keywords) > 1 else "experiment"

        if preferred_source == "pdf":
            file_name = chunk.file_name or f"{chunk.doc_id}.pdf"
            page_no = chunk.page_no or 1
            query = (
                f"In file '{file_name}' page {page_no}, what does the paper say about "
                f"{k1} and {k2}?"
            )
        else:
            title = chunk.title or chunk.doc_id
            query = f"Based on metadata, summarize the key idea of '{title}' with focus on {k1}."

        return EvalCase(
            query=query,
            ground_truth=chunk.text[:500],
            target_chunk_ids={chunk.chunk_id},
            expected_source=preferred_source,
        )

    @staticmethod
    def _extract_keywords(text: str, top_k: int = 3) -> list[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text)
        stop = {
            "this",
            "that",
            "with",
            "from",
            "paper",
            "using",
            "their",
            "which",
            "these",
            "results",
            "method",
        }
        uniq: list[str] = []
        seen = set()
        for token in tokens:
            t = token.lower()
            if t in stop or t in seen:
                continue
            seen.add(t)
            uniq.append(token)
            if len(uniq) >= top_k:
                break
        return uniq


# STEP_2_SUMMARY: Added synthetic query/testset generator for metadata and PDF ablation scenarios.
