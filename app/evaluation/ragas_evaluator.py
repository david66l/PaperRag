"""RAGAS evaluator that reuses the existing retrieval and generation pipelines."""

# OPTIMIZED_BY_CODEX_RAGAS_STEP_2
# FIXED_RAGAS_WITH_DASHSCOPE_STEP_1
# SWITCHED_TO_QWEN_TURBO_STEP_1
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
        self._last_token_usage: dict[str, int] = {"input_tokens_approx": 0, "rows": 0}  # SWITCHED_TO_QWEN_TURBO_STEP_2

    @property
    def judge_info(self) -> dict[str, str]:
        return {
            "base_url": self._judge["base_url"],
            "model": self._judge["model"],
            "embedding_model": self._judge["embedding_model"],
        }

    @property
    def last_token_usage(self) -> dict[str, int]:
        return dict(self._last_token_usage)

    def evaluate_variant(self, variant: AblationVariant, cases: list[EvalCase]) -> dict[str, float]:
        local_settings = self.settings.model_copy(deep=True)
        local_settings.rerank_enabled = variant.rerank_enabled
        retrieval_runtime_state = {
            "dense_disabled": os.getenv("PAPERRAG_EVAL_DISABLE_DENSE", "false").lower() in {"1", "true", "yes"}
        }  # SWITCHED_TO_QWEN_TURBO_STEP_17

        if variant.rerank_enabled and local_settings.reranker_provider == "none":
            local_settings.reranker_provider = "local"

        provider = create_embedding_provider(local_settings)
        retrieval_pipeline = RetrievalPipeline(local_settings, self.persistence, provider)
        generation_pipeline = GenerationPipeline(local_settings)
        context_builder = ContextBuilder(
            top_n=local_settings.top_n_context,
            max_tokens=local_settings.context_max_tokens,
        )
        if retrieval_runtime_state["dense_disabled"]:
            logger.warning("Dense retrieval is disabled for evaluation by PAPERRAG_EVAL_DISABLE_DENSE")  # SWITCHED_TO_QWEN_TURBO_STEP_18

        rows: list[dict[str, Any]] = []
        for case in cases:
            retrieval_result = self._run_with_retries(
                lambda: self._retrieve_with_rerank_fallback(
                    retrieval_pipeline=retrieval_pipeline,
                    query=case.query,
                    variant=variant,
                    local_settings=local_settings,
                    provider=provider,
                    runtime_state=retrieval_runtime_state,
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
        runtime_state: dict[str, bool],
    ) -> RetrievalResult:
        if runtime_state.get("dense_disabled"):
            self._disable_dense_path(retrieval_pipeline)
        try:
            return retrieval_pipeline.run(query, top_k=self.top_k)
        except Exception as exc:
            if self._is_embedding_bootstrap_failure(exc):
                logger.warning(
                    "Dense embedding bootstrap failed, switching to BM25-only retrieval for this variant: %s",
                    exc,
                )
                runtime_state["dense_disabled"] = True  # SWITCHED_TO_QWEN_TURBO_STEP_14
                self._disable_dense_path(retrieval_pipeline)
                return retrieval_pipeline.run(query, top_k=self.top_k)
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
        self._last_token_usage = self._estimate_token_usage(rows)  # SWITCHED_TO_QWEN_TURBO_STEP_3
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

        include_context_recall = os.getenv("PAPERRAG_ENABLE_CONTEXT_RECALL", "false").lower() in {"1", "true", "yes"}
        metric_plan = [
            ("faithfulness", faithfulness),
            ("context_precision", context_precision),
            ("answer_relevancy", answer_relevancy),
        ]
        if include_context_recall:
            metric_plan.append(("context_recall", context_recall))

        scores: dict[str, float] = {}
        for metric_name, metric_obj in metric_plan:
            kwargs: dict[str, Any] = {
                "metrics": [metric_obj],
                "llm": self._ragas_llm,
                "embeddings": self._ragas_embeddings,
            }
            try:
                try:
                    result = evaluate(dataset=dataset, **kwargs)
                except TypeError:
                    result = evaluate(dataset, **kwargs)
                scores[metric_name] = self._extract_metric_from_result(result, metric_name)
            except Exception as exc:
                logger.warning(
                    "RAGAS metric failed; fallback to 0.0 -> metric=%s model=%s base_url=%s error=%s",
                    metric_name,
                    self._judge["model"],
                    self._judge["base_url"],
                    exc,
                )
                scores[metric_name] = 0.0

        if "context_recall" not in scores:
            scores["context_recall"] = 0.0
        return scores

    def _extract_metric_from_result(self, result: Any, metric_name: str) -> float:
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            return self._extract_metric_from_dataframe(df, metric_name)
        if isinstance(result, dict):
            return self._extract_metric_from_dict(result, metric_name)
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
        model = os.getenv("PAPERRAG_RAGAS_JUDGE_MODEL") or os.getenv("PAPERRAG_LLM_MODEL") or "qwen-max"
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
            from langchain_openai import ChatOpenAI
            from langchain_openai import OpenAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper, llm_factory

            client = OpenAI(
                api_key=os.getenv("PAPERRAG_OPENAI_API_KEY", self._judge["api_key"]),  # SWITCHED_TO_QWEN_TURBO_STEP_5
                base_url=os.getenv("PAPERRAG_LLM_BASE_URL", self._judge["base_url"]),  # SWITCHED_TO_QWEN_TURBO_STEP_5
            )
            client = self._patch_dashscope_client(client)
            if "dashscope.aliyuncs.com" in self._judge["base_url"]:
                # DashScope works more reliably via plain ChatOpenAI text generations
                # than Instructor-structured calls in some RAGAS prompt paths.
                llm = LangchainLLMWrapper(
                    ChatOpenAI(
                        model=self._judge["model"],
                        api_key=self._judge["api_key"],
                        base_url=self._judge["base_url"],
                        temperature=0,
                    )
                )
            else:
                llm = llm_factory(
                    model=self._judge["model"],
                    provider="openai",
                    client=client,
                )
            logger.info(  # SWITCHED_TO_QWEN_TURBO_STEP_6
                "Initialized RAGAS judge runtime with model=%s base_url=%s",
                self._judge["model"],
                self._judge["base_url"],
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
        original_chat_create = client.chat.completions.create

        def safe_chat_create(*args, **kwargs):
            args, kwargs = self._normalize_request_payload(args, kwargs)
            self._log_message_shapes(kwargs.get("messages"))
            return original_chat_create(*args, **kwargs)

        client.chat.completions.create = safe_chat_create

        # Some OpenAI-compatible SDK paths use the Responses API with `input[*].content`.
        if hasattr(client, "responses") and hasattr(client.responses, "create"):
            original_responses_create = client.responses.create

            def safe_responses_create(*args, **kwargs):
                args, kwargs = self._normalize_request_payload(args, kwargs)
                self._log_message_shapes(kwargs.get("input"), is_input=True)
                return original_responses_create(*args, **kwargs)

            client.responses.create = safe_responses_create
        return client

    def _normalize_request_payload(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        normalized_kwargs = dict(kwargs)
        if "messages" in normalized_kwargs:
            normalized_kwargs["messages"] = self._normalize_messages(normalized_kwargs["messages"])
        if "input" in normalized_kwargs:
            normalized_kwargs["input"] = self._normalize_input(normalized_kwargs["input"])

        if args and isinstance(args[0], dict):
            first = dict(args[0])
            if "messages" in first:
                first["messages"] = self._normalize_messages(first["messages"])
            if "input" in first:
                first["input"] = self._normalize_input(first["input"])
            normalized_args = (first, *args[1:])
        else:
            normalized_args = args
        return normalized_args, normalized_kwargs

    def _normalize_messages(self, messages: Any) -> Any:
        if not isinstance(messages, list):
            return messages
        normalized: list[Any] = []
        for message in messages:
            if not isinstance(message, dict):
                normalized.append(message)
                continue
            message_copy = dict(message)
            if "content" in message_copy:
                message_copy["content"] = self._normalize_text_content(message_copy.get("content"))
            normalized.append(message_copy)
        has_user = any(isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user" for msg in normalized)
        if not has_user:
            seed_text = ""
            for msg in normalized:
                if isinstance(msg, dict) and msg.get("content"):
                    seed_text = str(msg.get("content"))
                    break
            normalized.append({"role": "user", "content": seed_text or "Please follow the instructions and answer."})
        return normalized

    def _normalize_input(self, payload: Any) -> Any:
        if isinstance(payload, list):
            normalized_items: list[Any] = []
            for item in payload:
                if not isinstance(item, dict):
                    normalized_items.append(item)
                    continue
                item_copy = dict(item)
                if "content" in item_copy:
                    item_copy["content"] = self._normalize_text_content(item_copy.get("content"))
                normalized_items.append(item_copy)
            return normalized_items
        if isinstance(payload, dict):
            payload_copy = dict(payload)
            if "content" in payload_copy:
                payload_copy["content"] = self._normalize_text_content(payload_copy.get("content"))
            return payload_copy
        return payload

    def _normalize_text_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # DashScope-compatible mode expects string-like content; flatten list into text.
            parts = [self._normalize_text_content(item) for item in content]
            parts = [part for part in parts if part]
            return "\n".join(parts)
        if isinstance(content, dict):
            for key in ("text", "input_text", "content"):
                if key in content:
                    return self._normalize_text_content(content.get(key))
            return str(content)
        return str(content)

    def _log_message_shapes(self, payload: Any, is_input: bool = False) -> None:
        if not isinstance(payload, list):
            return
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = item.get("content", "")
            else:
                role = "user"
                content = str(item)
            content_len = len(str(content))
            logger.info(
                "DashScope judge message -> model=%s source=%s idx=%s role=%s content_len=%s",
                self._judge["model"],
                "input" if is_input else "messages",
                index,
                role,
                content_len,
            )

    def _extract_metric_from_dataframe(self, dataframe, metric: str) -> float:
        # SWITCHED_TO_QWEN_TURBO_STEP_7
        if metric not in dataframe:
            raise RuntimeError(f"RAGAS output missing metric column: {metric}")
        series = dataframe[metric].dropna()
        if series.empty:
            raise RuntimeError(f"RAGAS output metric is empty: {metric}")
        return float(series.mean())

    def _extract_metric_from_dict(self, payload: dict[str, Any], metric: str) -> float:
        # SWITCHED_TO_QWEN_TURBO_STEP_7
        if metric not in payload:
            raise RuntimeError(f"RAGAS output missing metric field: {metric}")
        value = payload[metric]
        if value is None:
            raise RuntimeError(f"RAGAS output metric is None: {metric}")
        return float(value)

    def _estimate_token_usage(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        # SWITCHED_TO_QWEN_TURBO_STEP_8
        total_chars = 0
        for row in rows:
            total_chars += len(str(row.get("question", "")))
            total_chars += len(str(row.get("answer", "")))
            total_chars += len(str(row.get("ground_truth", "")))
            total_chars += sum(len(str(ctx)) for ctx in row.get("contexts", []))
        return {
            "input_tokens_approx": max(1, total_chars // 4),
            "rows": len(rows),
        }

    def _disable_dense_path(self, retrieval_pipeline: RetrievalPipeline) -> None:
        # SWITCHED_TO_QWEN_TURBO_STEP_15
        retrieval_pipeline.dense_retriever.retrieve = lambda _query, _top_k: []

    def _is_embedding_bootstrap_failure(self, exc: Exception) -> bool:
        # SWITCHED_TO_QWEN_TURBO_STEP_16
        message = str(exc).lower()
        return any(
            token in message
            for token in [
                "sentence-transformers",
                "huggingface",
                "cannot send a request, as the client has been closed",
                "ssl: unexpected eof while reading",
                "adapter_config.json",
                "modules.json",
            ]
        )


# STEP_2_SUMMARY: Added RAGAS evaluator that runs 4 ablation variants using existing retrieval + generation pipelines.
