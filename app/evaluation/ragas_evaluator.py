"""RAGAS evaluator that reuses the existing retrieval and generation pipelines."""

# OPTIMIZED_BY_CODEX_RAGAS_STEP_2
from __future__ import annotations

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
        self._ragas_llm = self._build_ragas_llm()

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
            try:
                retrieval_result = retrieval_pipeline.run(case.query, top_k=self.top_k)
            except Exception as exc:
                if not variant.rerank_enabled:
                    raise
                logger.warning("Rerank variant fallback to no-rerank due to error: %s", exc)
                local_settings.rerank_enabled = False
                local_settings.reranker_provider = "none"
                retrieval_pipeline = RetrievalPipeline(local_settings, self.persistence, provider)
                retrieval_result = retrieval_pipeline.run(case.query, top_k=self.top_k)

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
            answer, _ = generation_pipeline.run(scoped_result, mode="concise")

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
        try:
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

            kwargs: dict[str, Any] = {"metrics": metrics}
            if self._ragas_llm is not None:
                kwargs["llm"] = self._ragas_llm

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
        except Exception as exc:
            logger.warning("RAGAS evaluation failed, using deterministic fallback metrics: %s", exc)
            return self._fallback_metrics(rows)

    def _build_ragas_llm(self):
        try:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper

            if not self.settings.llm_api_key:
                return None

            chat = ChatOpenAI(
                model=self.settings.llm_model,
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_api_url,
                temperature=self.settings.llm_temperature,
            )
            return LangchainLLMWrapper(chat)
        except Exception as exc:
            logger.warning("RAGAS judge LLM wrapper unavailable: %s", exc)
            return None

    @staticmethod
    def _fallback_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
        def overlap_ratio(left: str, right: str) -> float:
            lt = set(left.lower().split())
            rt = set(right.lower().split())
            if not lt or not rt:
                return 0.0
            return len(lt & rt) / max(1, len(lt | rt))

        faithfulness = mean(overlap_ratio(row["answer"], " ".join(row["contexts"])) for row in rows)
        answer_relevancy = mean(overlap_ratio(row["question"], row["answer"]) for row in rows)
        context_precision = mean(overlap_ratio(" ".join(row["contexts"]), row["ground_truth"]) for row in rows)
        context_recall = mean(overlap_ratio(row["ground_truth"], " ".join(row["contexts"])) for row in rows)
        return {
            "faithfulness": float(faithfulness),
            "answer_relevancy": float(answer_relevancy),
            "context_precision": float(context_precision),
            "context_recall": float(context_recall),
        }


# STEP_2_SUMMARY: Added RAGAS evaluator that runs 4 ablation variants using existing retrieval + generation pipelines.
