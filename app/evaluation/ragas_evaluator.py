"""RAGAS evaluator that reuses the existing retrieval and generation pipelines."""

# OPTIMIZED_BY_CODEX_RAGAS_STEP_2
# FIXED_RAGAS_WITH_DASHSCOPE_STEP_1
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.schemas import RetrievalResult
from app.embedding.pipeline import create_embedding_provider
from app.evaluation.retrieval_eval import recall_at_k
from app.evaluation.testset_generator import EvalCase
from app.generation.pipeline import GenerationPipeline
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.pipeline import RetrievalPipeline
from app.storage.persistence import PersistenceManager

logger = get_logger(__name__)


@dataclass
class AblationVariant:
    """One ablation variant configuration."""

    name: str
    allowed_sources: set[str]
    rerank_enabled: bool


class RagasAblationEvaluator:
    """Run ablation groups and evaluate with standard RAGAS metrics."""

    def __init__(
        self,
        settings: Settings,
        persistence: PersistenceManager,
        top_k: int = 30,
    ):
        self.settings = settings
        self.persistence = persistence
        self.top_k = top_k
        self._judge = self._resolve_judge_config()
        self._ragas_llm, self._ragas_embeddings = self._build_ragas_runtime()

    @property
    def judge_info(self) -> dict[str, str]:
        return {
            "base_url": self._judge["base_url"],
            "model": self._judge["model"],
            "embedding_model": self._judge["embedding_model"],
        }

    def evaluate_variant(self, variant: AblationVariant, cases: list[EvalCase]) -> dict[str, float]:
        local_settings = self.settings.model_copy(deep=True)
        local_settings.rerank_enabled = variant.rerank_enabled

        if variant.rerank_enabled and local_settings.reranker_provider == "none":
            local_settings.reranker_provider = "local"

        provider = create_embedding_provider(local_settings)
        retrieval_pipeline = RetrievalPipeline(local_settings, self.persistence, provider)
        generation_pipeline = GenerationPipeline(local_settings)
        context_builder = ContextBuilder(
            top_n=local_settings.top_n_context,
            max_tokens=local_settings.context_max_tokens,
        )

        rows: list[dict[str, Any]] = []
        for case in cases:
            retrieval_result = self._run_with_retries(
                lambda: self._retrieve_with_rerank_fallback(
                    retrieval_pipeline=retrieval_pipeline,
                    query=case.query,
                    variant=variant,
                    local_settings=local_settings,
                    provider=provider,
                ),
                operation=f"retrieve:{case.query[:48]}",
            )

            filtered_candidates = [
                candidate for candidate in retrieval_result.candidates if candidate.source_type in variant.allowed_sources
            ]

            context_text, citations = context_builder.build(filtered_candidates)
            citation_text_map = {candidate.chunk_id: candidate.text for candidate in filtered_candidates}
            contexts = [citation_text_map.get(citation.chunk_id, "") for citation in citations if citation.chunk_id in citation_text_map]

            scoped_result = RetrievalResult(
                query=case.query,
                candidates=filtered_candidates,
                context_text=context_text,
                citations=citations,
                trace=retrieval_result.trace,
            )
            answer, _ = self._run_with_retries(
                lambda: generation_pipeline.run(scoped_result, mode="concise"),
                operation=f"generate:{case.query[:48]}",
            )

            retrieved_ids = [candidate.chunk_id for candidate in filtered_candidates]
            row = {
                "question": case.query,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": case.ground_truth,
                "recall_at_5": recall_at_k(case.target_chunk_ids, retrieved_ids, 5),
            }
            rows.append(row)

        return self._evaluate_rows_with_ragas(rows)

    # FIXED_RAGAS_WITH_DASHSCOPE_STEP_3
    def _run_with_retries(self, fn, operation: str, retries: int = 3, sleep_seconds: float = 2.0):
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt >= retries:
                    break
                message = str(exc).lower()
                transient = any(token in message for token in ["ssl", "timeout", "connect", "eof"])
                if not transient:
                    raise
                logger.warning(
                    "Transient error on %s (attempt %s/%s): %s",
                    operation,
                    attempt,
                    retries,
                    exc,
                )
                time.sleep(sleep_seconds)
        raise RuntimeError(f"{operation} failed after {retries} attempts: {last_exc}") from last_exc

    def _retrieve_with_rerank_fallback(
        self,
        retrieval_pipeline: RetrievalPipeline,
        query: str,
        variant: AblationVariant,
        local_settings: Settings,
        provider,
    ) -> RetrievalResult:
        try:
            return retrieval_pipeline.run(query, top_k=self.top_k)
        except Exception as exc:
            if not variant.rerank_enabled:
                raise
            logger.warning("Rerank variant fallback to no-rerank due to error: %s", exc)
            local_settings.rerank_enabled = False
            local_settings.reranker_provider = "none"
            fallback_pipeline = RetrievalPipeline(local_settings, self.persistence, provider)
            return fallback_pipeline.run(query, top_k=self.top_k)

    def _evaluate_rows_with_ragas(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "recall_at_5": 0.0,
            }

        ragas_scores = self._run_ragas(rows)
        ragas_scores["recall_at_5"] = float(mean(float(row.get("recall_at_5", 0.0)) for row in rows))
        return ragas_scores

    def _run_ragas(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict(
            {
                "question": [row["question"] for row in rows],
                "answer": [row["answer"] for row in rows],
                "contexts": [row["contexts"] for row in rows],
                "ground_truth": [row["ground_truth"] for row in rows],
            }
        )

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        kwargs: dict[str, Any] = {
            "metrics": metrics,
            "llm": self._ragas_llm,
            "embeddings": self._ragas_embeddings,
        }

        try:
            result = evaluate(dataset=dataset, **kwargs)
        except TypeError:
            result = evaluate(dataset, **kwargs)

        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            return {
                "faithfulness": float(df["faithfulness"].fillna(0.0).mean()) if "faithfulness" in df else 0.0,
                "answer_relevancy": float(df["answer_relevancy"].fillna(0.0).mean()) if "answer_relevancy" in df else 0.0,
                "context_precision": float(df["context_precision"].fillna(0.0).mean()) if "context_precision" in df else 0.0,
                "context_recall": float(df["context_recall"].fillna(0.0).mean()) if "context_recall" in df else 0.0,
            }

        if isinstance(result, dict):
            return {
                "faithfulness": float(result.get("faithfulness", 0.0)),
                "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
                "context_precision": float(result.get("context_precision", 0.0)),
                "context_recall": float(result.get("context_recall", 0.0)),
            }

        raise RuntimeError("Unsupported RAGAS result type")

    def _resolve_judge_config(self) -> dict[str, str]:
        api_key = (
            os.getenv("PAPERRAG_OPENAI_API_KEY")
            or os.getenv("PAPERRAG_LLM_API_KEY")
            or self.settings.llm_api_key
        )
        base_url = (
            os.getenv("PAPERRAG_LLM_BASE_URL")
            or os.getenv("PAPERRAG_OPENAI_BASE_URL")
            or self.settings.llm_api_url
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        model = os.getenv("PAPERRAG_LLM_MODEL") or self.settings.llm_model or "qwen3-8b"
        embedding_model = os.getenv("PAPERRAG_RAGAS_EMBEDDING_MODEL") or "text-embedding-v3"

        if not api_key:
            raise RuntimeError(
                "RAGAS judge requires API key. Set PAPERRAG_OPENAI_API_KEY or PAPERRAG_LLM_API_KEY."
            )

        return {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "embedding_model": embedding_model,
        }

    def _build_ragas_runtime(self):
        try:
            from openai import OpenAI
            from langchain_openai import OpenAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import llm_factory

            client = OpenAI(
                api_key=self._judge["api_key"],
                base_url=self._judge["base_url"],
            )
            client = self._patch_dashscope_client(client)
            llm = llm_factory(
                model=self._judge["model"],
                provider="openai",
                client=client,
            )
            embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    model=self._judge["embedding_model"],
                    api_key=self._judge["api_key"],
                    base_url=self._judge["base_url"],
                )
            )
            return llm, embeddings
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize RAGAS judge runtime: {exc}") from exc

    def _patch_dashscope_client(self, client):
        # FIXED_RAGAS_WITH_DASHSCOPE_STEP_4
        original_create = client.chat.completions.create

        def safe_create(*args, **kwargs):
            messages = kwargs.get("messages")
            if isinstance(messages, list):
                normalized = []
                for message in messages:
                    if not isinstance(message, dict):
                        normalized.append(message)
                        continue
                    content = message.get("content")
                    if isinstance(content, list):
                        parts: list[str] = []
                        for item in content:
                            if isinstance(item, str):
                                parts.append(item)
                            elif isinstance(item, dict):
                                text = (
                                    item.get("text")
                                    or item.get("input_text")
                                    or item.get("content")
                                    or ""
                                )
                                if text:
                                    parts.append(str(text))
                            else:
                                parts.append(str(item))
                        msg_copy = dict(message)
                        msg_copy["content"] = "\n".join(parts)
                        normalized.append(msg_copy)
                    else:
                        normalized.append(message)
                kwargs["messages"] = normalized
            return original_create(*args, **kwargs)

        client.chat.completions.create = safe_create
        return client


# STEP_2_SUMMARY: Added RAGAS evaluator that runs 4 ablation variants using existing retrieval + generation pipelines.
