"""Generation evaluation powered by official RAGAS metrics."""

# OPTIMIZED_BY_CODEX_RAGAS_STEP_2
from __future__ import annotations

from statistics import mean
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class GenerationEvaluator:
    """Evaluate generation quality using RAGAS (with fallback)."""

    def evaluate(self, test_cases: list[dict[str, Any]]) -> dict[str, float]:
        if not test_cases:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
            }

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, faithfulness

            dataset = Dataset.from_dict(
                {
                    "question": [row.get("question", "") for row in test_cases],
                    "answer": [row.get("answer", "") for row in test_cases],
                    "contexts": [row.get("contexts", []) for row in test_cases],
                    "ground_truth": [row.get("ground_truth", "") for row in test_cases],
                }
            )
            result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy, context_precision])
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                return {
                    "faithfulness": float(df["faithfulness"].fillna(0.0).mean()) if "faithfulness" in df else 0.0,
                    "answer_relevancy": (
                        float(df["answer_relevancy"].fillna(0.0).mean()) if "answer_relevancy" in df else 0.0
                    ),
                    "context_precision": (
                        float(df["context_precision"].fillna(0.0).mean()) if "context_precision" in df else 0.0
                    ),
                }
            if isinstance(result, dict):
                return {
                    "faithfulness": float(result.get("faithfulness", 0.0)),
                    "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
                    "context_precision": float(result.get("context_precision", 0.0)),
                }
            raise RuntimeError("Unsupported RAGAS result type")
        except Exception as exc:
            logger.warning("GenerationEvaluator fallback metrics due to RAGAS error: %s", exc)
            return self._fallback(test_cases)

    @staticmethod
    def _fallback(test_cases: list[dict[str, Any]]) -> dict[str, float]:
        def overlap_ratio(left: str, right: str) -> float:
            lt = set(left.lower().split())
            rt = set(right.lower().split())
            if not lt or not rt:
                return 0.0
            return len(lt & rt) / max(1, len(lt | rt))

        faithfulness = mean(overlap_ratio(row.get("answer", ""), " ".join(row.get("contexts", []))) for row in test_cases)
        answer_relevancy = mean(overlap_ratio(row.get("question", ""), row.get("answer", "")) for row in test_cases)
        context_precision = mean(
            overlap_ratio(" ".join(row.get("contexts", [])), row.get("ground_truth", "")) for row in test_cases
        )
        return {
            "faithfulness": float(faithfulness),
            "answer_relevancy": float(answer_relevancy),
            "context_precision": float(context_precision),
        }


# STEP_2_SUMMARY: generation_eval now computes official RAGAS generation metrics with deterministic fallback.
