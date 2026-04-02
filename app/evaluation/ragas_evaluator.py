"""RAGAS-backed evaluator with robust local fallback metrics."""

# OPTIMIZED_BY_CODEX_STEP_2
from __future__ import annotations

import re
from statistics import mean
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class RagasEvaluator:
    """Evaluate quality metrics for one ablation group."""

    def __init__(self, use_ragas: bool = True):
        self.use_ragas = use_ragas

    def evaluate(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "recall_at_5": 0.0,
            }

        recall_at_5 = mean(float(row.get("recall_at_5", 0.0)) for row in rows)

        ragas_scores: dict[str, float] | None = None
        if self.use_ragas:
            ragas_scores = self._evaluate_with_ragas(rows)

        if ragas_scores is None:
            ragas_scores = self._evaluate_with_fallback(rows)

        ragas_scores["recall_at_5"] = float(recall_at_5)
        return ragas_scores

    def _evaluate_with_ragas(self, rows: list[dict[str, Any]]) -> dict[str, float] | None:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, faithfulness

            data = {
                "question": [row["question"] for row in rows],
                "answer": [row["answer"] for row in rows],
                "contexts": [row["contexts"] for row in rows],
                "ground_truth": [row["ground_truth"] for row in rows],
            }
            ds = Dataset.from_dict(data)
            result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
            df = result.to_pandas()

            faithfulness_score = float(df["faithfulness"].fillna(0.0).mean()) if "faithfulness" in df else 0.0
            answer_relevancy_score = (
                float(df["answer_relevancy"].fillna(0.0).mean()) if "answer_relevancy" in df else 0.0
            )
            context_precision_score = (
                float(df["context_precision"].fillna(0.0).mean()) if "context_precision" in df else 0.0
            )

            return {
                "faithfulness": faithfulness_score,
                "answer_relevancy": answer_relevancy_score,
                "context_precision": context_precision_score,
            }
        except Exception as exc:
            logger.warning("RAGAS evaluation fallback triggered: %s", exc)
            return None

    def _evaluate_with_fallback(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        faithfulness_scores: list[float] = []
        relevancy_scores: list[float] = []
        context_precision_scores: list[float] = []

        for row in rows:
            question_tokens = self._tokenize(row["question"])
            answer_tokens = self._tokenize(row["answer"])
            ctx_tokens = self._tokenize(" ".join(row["contexts"]))
            gt_tokens = self._tokenize(row["ground_truth"])

            faithfulness_scores.append(self._jaccard(answer_tokens, ctx_tokens))
            relevancy_scores.append(self._jaccard(question_tokens, answer_tokens))

            if row["contexts"]:
                per_ctx = [
                    self._jaccard(self._tokenize(context), gt_tokens if gt_tokens else question_tokens)
                    for context in row["contexts"]
                ]
                context_precision_scores.append(mean(per_ctx))
            else:
                context_precision_scores.append(0.0)

        return {
            "faithfulness": float(mean(faithfulness_scores)),
            "answer_relevancy": float(mean(relevancy_scores)),
            "context_precision": float(mean(context_precision_scores)),
        }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9\-]{2,}", text.lower()))

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / max(1, len(a | b))


# STEP_2_SUMMARY: Added RAGAS evaluator with automatic local fallback for stable metric computation.
