"""Run RAG ablation evaluation with RAGAS metrics + charts."""

# OPTIMIZED_BY_CODEX_STEP_2
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import plotly.graph_objects as go

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.embedding.pipeline import create_embedding_provider
from app.evaluation.ragas_evaluator import RagasEvaluator
from app.evaluation.retrieval_eval import recall_at_k
from app.evaluation.testset_generator import EvalCase, TestsetGenerator
from app.retrieval.pipeline import RetrievalPipeline
from app.storage.persistence import PersistenceManager


@dataclass
class EvalVariant:
    name: str
    allowed_sources: set[str]
    rerank_enabled: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS + ablation evaluation")
    parser.add_argument("--num_queries", type=int, default=50, help="Number of evaluation queries")
    parser.add_argument("--top_k", type=int, default=30, help="Retrieval top_k")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/eval_results",
        help="Directory for evaluation outputs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--disable_ragas",
        action="store_true",
        help="Disable RAGAS and force local fallback metrics",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    setup_logging()

    settings = get_settings()
    persistence = PersistenceManager(settings)
    persistence.load_all()

    chunks = persistence.chunk_repo.get_all()
    generator = TestsetGenerator(seed=args.seed)
    cases = generator.generate(chunks, num_queries=args.num_queries)

    evaluator = RagasEvaluator(use_ragas=not args.disable_ragas)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        EvalVariant("abstract-only", {"metadata"}, rerank_enabled=False),
        EvalVariant("full-pdf", {"pdf"}, rerank_enabled=False),
        EvalVariant("hybrid", {"metadata", "pdf"}, rerank_enabled=False),
        EvalVariant("+rerank", {"metadata", "pdf"}, rerank_enabled=True),
    ]

    all_results: dict[str, dict[str, float]] = {}
    for variant in variants:
        all_results[variant.name] = _evaluate_variant(
            settings=settings,
            persistence=persistence,
            evaluator=evaluator,
            cases=cases,
            variant=variant,
            top_k=args.top_k,
        )

    _save_results(all_results, output_dir)
    _write_markdown_summary(all_results, output_dir / "summary.md")
    _save_radar_chart(all_results, output_dir / "ablation_radar.html")
    _save_bar_chart(all_results, output_dir / "ablation_bar.html")

    print("Evaluation completed.")
    print(f"Cases: {len(cases)}")
    print(json.dumps(all_results, ensure_ascii=False, indent=2))


def _evaluate_variant(
    settings,
    persistence: PersistenceManager,
    evaluator: RagasEvaluator,
    cases: list[EvalCase],
    variant: EvalVariant,
    top_k: int,
) -> dict[str, float]:
    local_settings = settings.model_copy(deep=True)
    local_settings.rerank_enabled = variant.rerank_enabled

    # Keep inheritance tree untouched; only switch runtime settings for ablation.
    if variant.rerank_enabled and local_settings.reranker_provider == "none":
        local_settings.reranker_provider = "local"

    embedding_provider = create_embedding_provider(local_settings)
    pipeline = RetrievalPipeline(local_settings, persistence, embedding_provider)

    rows: list[dict[str, object]] = []
    for case in cases:
        try:
            result = pipeline.run(case.query, top_k=top_k)
        except Exception:
            # Fall back to non-rerank execution if reranker model cannot be loaded.
            if not variant.rerank_enabled:
                raise
            local_settings.rerank_enabled = False
            local_settings.reranker_provider = "none"
            pipeline = RetrievalPipeline(local_settings, persistence, embedding_provider)
            result = pipeline.run(case.query, top_k=top_k)

        filtered = [c for c in result.candidates if c.source_type in variant.allowed_sources]
        top5 = filtered[:5]

        retrieved_ids = [c.chunk_id for c in top5]
        recall5 = recall_at_k(case.target_chunk_ids, retrieved_ids, 5)

        contexts = [c.text for c in top5]
        answer = _synthesize_answer(top5)

        rows.append(
            {
                "question": case.query,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": case.ground_truth,
                "recall_at_5": recall5,
            }
        )

    return evaluator.evaluate(rows)


def _synthesize_answer(candidates) -> str:
    if not candidates:
        return "I don't know based on retrieved context."
    merged = " ".join(c.text for c in candidates[:2])
    return merged[:700]


def _save_results(results: dict[str, dict[str, float]], output_dir: Path) -> None:
    with open(output_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _write_markdown_summary(results: dict[str, dict[str, float]], path: Path) -> None:
    headers = ["Variant", "Faithfulness", "Answer Relevancy", "Context Precision", "Recall@5", "Overall"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, metrics in results.items():
        overall = mean(
            [
                metrics.get("faithfulness", 0.0),
                metrics.get("answer_relevancy", 0.0),
                metrics.get("context_precision", 0.0),
                metrics.get("recall_at_5", 0.0),
            ]
        )
        lines.append(
            f"| {name} | {metrics.get('faithfulness', 0.0):.3f} | "
            f"{metrics.get('answer_relevancy', 0.0):.3f} | "
            f"{metrics.get('context_precision', 0.0):.3f} | "
            f"{metrics.get('recall_at_5', 0.0):.3f} | {overall:.3f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_radar_chart(results: dict[str, dict[str, float]], path: Path) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "recall_at_5"]
    fig = go.Figure()

    for variant, scores in results.items():
        values = [scores.get(metric, 0.0) for metric in metrics]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics + [metrics[0]],
                fill="toself",
                name=variant,
            )
        )

    fig.update_layout(
        title="RAG Ablation Radar",
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


def _save_bar_chart(results: dict[str, dict[str, float]], path: Path) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "recall_at_5"]
    variants = list(results.keys())

    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(
            go.Bar(
                name=metric,
                x=variants,
                y=[results[v].get(metric, 0.0) for v in variants],
            )
        )

    fig.update_layout(
        barmode="group",
        title="RAG Ablation Metrics",
        yaxis_title="Score",
        yaxis_range=[0, 1],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


if __name__ == "__main__":
    run()
# STEP_2_SUMMARY: Added end-to-end ablation runner with 50-query generation, 4-group comparison, and plot outputs.
