"""Generate evaluation test cases from indexed metadata/PDF chunks."""

# OPTIMIZED_BY_CODEX_RAGAS_STEP_2
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from app.core.schemas import Chunk


@dataclass
class EvalCase:
    """One query case for RAGAS evaluation."""

    query: str
    ground_truth: str
    target_chunk_ids: set[str]
    expected_source: str


class TestsetGenerator:
    """Generate testset queries with metadata-first strategy."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, chunks: list[Chunk], num_queries: int = 50) -> list[EvalCase]:
        if not chunks:
            return []

        metadata_chunks = [chunk for chunk in chunks if chunk.source_type == "metadata"]
        pdf_chunks = [chunk for chunk in chunks if chunk.source_type == "pdf"]

        base_pool = metadata_chunks if metadata_chunks else chunks
        cases = self._sample_cases(base_pool, max(1, int(num_queries * 0.7)), preferred_source="metadata")

        if pdf_chunks:
            pdf_target = num_queries - len(cases)
            cases.extend(self._sample_cases(pdf_chunks, max(0, pdf_target), preferred_source="pdf"))

        while len(cases) < num_queries:
            chosen = self._rng.choice(chunks)
            cases.append(self._build_case(chosen, preferred_source=chosen.source_type))

        self._rng.shuffle(cases)
        return cases[:num_queries]

    def _sample_cases(self, pool: list[Chunk], count: int, preferred_source: str) -> list[EvalCase]:
        if not pool or count <= 0:
            return []

        if len(pool) >= count:
            selected = self._rng.sample(pool, count)
        else:
            selected = [self._rng.choice(pool) for _ in range(count)]

        return [self._build_case(chunk, preferred_source=preferred_source) for chunk in selected]

    def _build_case(self, chunk: Chunk, preferred_source: str) -> EvalCase:
        keywords = self._extract_keywords(chunk.text)
        key_phrase = ", ".join(keywords[:2]) if keywords else "core contribution"

        if preferred_source == "pdf":
            file_name = chunk.file_name or f"{chunk.doc_id}.pdf"
            page_no = chunk.page_no or 1
            query = (
                f"Based only on PDF file '{file_name}' page {page_no}, explain the method details related to {key_phrase}."
            )
        else:
            title = chunk.title or chunk.doc_id
            categories = ", ".join(chunk.categories[:2]) if chunk.categories else "AI research"
            query = (
                f"For paper '{title}' in categories {categories}, summarize the main idea and evidence about {key_phrase}."
            )

        return EvalCase(
            query=query,
            ground_truth=chunk.text[:700],
            target_chunk_ids={chunk.chunk_id},
            expected_source=preferred_source,
        )

    @staticmethod
    def _extract_keywords(text: str, top_k: int = 5) -> list[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text)
        stop_words = {
            "paper",
            "method",
            "results",
            "using",
            "their",
            "this",
            "that",
            "with",
            "from",
            "model",
            "models",
        }

        uniq: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            lowered = token.lower()
            if lowered in stop_words or lowered in seen:
                continue
            seen.add(lowered)
            uniq.append(token)
            if len(uniq) >= top_k:
                break
        return uniq


# STEP_2_SUMMARY: Testset generator now creates 50-query metadata/PDF mixed cases for RAGAS ablation.
