"""Retrieval evaluation metrics with RAGAS context scoring."""

# OPTIMIZED_BY_CODEX_RAGAS_STEP_2

from typing import Any


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Recall@k: fraction of relevant items in top-k."""
    top_k = set(retrieved[:k])
    if not relevant:
        return 0.0
    return len(relevant & top_k) / len(relevant)


def mrr(relevant: set[str], retrieved: list[str]) -> float:
    """Mean Reciprocal Rank."""
    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain @k (simplified)."""
    import math
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


class RetrievalEvaluator:
    """Evaluate retrieval quality with Recall@k + RAGAS context metrics."""

    def evaluate(self, queries: list[dict[str, Any]]) -> dict[str, float]:
        if not queries:
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "recall_at_5": 0.0,
                "mrr": 0.0,
            }

        recall5_scores: list[float] = []
        mrr_scores: list[float] = []
        for row in queries:
            relevant = set(row.get("relevant_ids", []))
            retrieved = row.get("retrieved_ids", [])
            recall5_scores.append(recall_at_k(relevant, retrieved, 5))
            mrr_scores.append(mrr(relevant, retrieved))

        ragas_scores = self._evaluate_context_metrics(queries)
        ragas_scores["recall_at_5"] = sum(recall5_scores) / len(recall5_scores)
        ragas_scores["mrr"] = sum(mrr_scores) / len(mrr_scores)
        return ragas_scores

    @staticmethod
    def _evaluate_context_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import context_precision, context_recall

            dataset = Dataset.from_dict(
                {
                    "question": [row.get("question", "") for row in rows],
                    "answer": [" ".join(row.get("contexts", [])[:1]) for row in rows],
                    "contexts": [row.get("contexts", []) for row in rows],
                    "ground_truth": [row.get("ground_truth", "") for row in rows],
                }
            )
            result = evaluate(dataset=dataset, metrics=[context_precision, context_recall])
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                return {
                    "context_precision": (
                        float(df["context_precision"].fillna(0.0).mean()) if "context_precision" in df else 0.0
                    ),
                    "context_recall": float(df["context_recall"].fillna(0.0).mean()) if "context_recall" in df else 0.0,
                }
            if isinstance(result, dict):
                return {
                    "context_precision": float(result.get("context_precision", 0.0)),
                    "context_recall": float(result.get("context_recall", 0.0)),
                }
            raise RuntimeError("Unsupported RAGAS result type")
        except Exception:
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
            }


# STEP_2_SUMMARY: retrieval_eval now supports RAGAS context metrics plus Recall@5 and MRR aggregation.
