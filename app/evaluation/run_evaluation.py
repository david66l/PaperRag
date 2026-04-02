"""Run RAGAS-based ablation evaluation and generate visualization artifacts."""

# OPTIMIZED_BY_CODEX_RAGAS_STEP_2
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import plotly.graph_objects as go

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.schemas import BuildIndexRequest
from app.evaluation.ragas_evaluator import AblationVariant, RagasAblationEvaluator
from app.evaluation.testset_generator import TestsetGenerator
from app.services.index_service import IndexService
from app.storage.persistence import PersistenceManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation with ablation")
    parser.add_argument("--num_queries", type=int, default=50, help="Number of evaluation queries")
    parser.add_argument("--top_k", type=int, default=30, help="Retrieval top_k for ablation")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/eval_results",
        help="Output directory for JSON/Markdown/plots",
    )
    parser.add_argument("--max_papers", type=int, default=0, help="Build index if empty using this metadata limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for testset generation")
    parser.add_argument("--data_path", type=str, default="", help="Optional metadata path for cold-start indexing")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    setup_logging()

    settings = get_settings()
    persistence = PersistenceManager(settings)
    _ensure_index_ready(settings, persistence, max_papers=args.max_papers, data_path=args.data_path)

    chunks = persistence.chunk_repo.get_all()
    cases = TestsetGenerator(seed=args.seed).generate(chunks, num_queries=args.num_queries)

    variants = [
        AblationVariant(name="abstract-only", allowed_sources={"metadata"}, rerank_enabled=False),
        AblationVariant(name="full-pdf", allowed_sources={"pdf"}, rerank_enabled=False),
        AblationVariant(name="hybrid", allowed_sources={"metadata", "pdf"}, rerank_enabled=False),
        AblationVariant(name="+rerank", allowed_sources={"metadata", "pdf"}, rerank_enabled=True),
    ]

    evaluator = RagasAblationEvaluator(settings=settings, persistence=persistence, top_k=args.top_k)

    results: dict[str, dict[str, float]] = {}
    for variant in variants:
        results[variant.name] = evaluator.evaluate_variant(variant, cases)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_json(results, output_dir / "ablation_results.json")
    _save_summary_markdown(results, output_dir / "summary.md")
    _save_radar_chart(results, output_dir / "ablation_radar.html")
    _save_bar_chart(results, output_dir / "ablation_bar.html")

    print("Evaluation completed")
    print(f"Queries: {len(cases)}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


def _ensure_index_ready(settings, persistence: PersistenceManager, max_papers: int, data_path: str) -> None:
    try:
        persistence.load_all()
    except Exception:
        pass

    if persistence.is_ready():
        return

    if max_papers <= 0:
        raise RuntimeError(
            "Index is not ready. Build index first or pass --max_papers to allow evaluation bootstrap build."
        )

    service = IndexService(settings, persistence)
    metadata_path = data_path or str(settings.abs_data_dir / settings.default_data_file)
    request = BuildIndexRequest(
        data_path=metadata_path,
        limit=max_papers,
        rebuild=True,
    )
    result = service.build(request)
    if result.status != "success":
        raise RuntimeError(f"Index bootstrap failed: {result.message}")


def _save_json(results: dict[str, dict[str, float]], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


# OPTIMIZED_BY_CODEX_RAGAS_STEP_3
def _save_summary_markdown(results: dict[str, dict[str, float]], output_path: Path) -> None:
    headers = [
        "Variant",
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision",
        "Context Recall",
        "Recall@5",
        "Overall",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for variant, metrics in results.items():
        overall = mean(
            [
                metrics.get("faithfulness", 0.0),
                metrics.get("answer_relevancy", 0.0),
                metrics.get("context_precision", 0.0),
                metrics.get("context_recall", 0.0),
            ]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    f"{metrics.get('faithfulness', 0.0):.3f}",
                    f"{metrics.get('answer_relevancy', 0.0):.3f}",
                    f"{metrics.get('context_precision', 0.0):.3f}",
                    f"{metrics.get('context_recall', 0.0):.3f}",
                    f"{metrics.get('recall_at_5', 0.0):.3f}",
                    f"{overall:.3f}",
                ]
            )
            + " |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_radar_chart(results: dict[str, dict[str, float]], output_path: Path) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    figure = go.Figure()

    for variant, values in results.items():
        radial_values = [values.get(metric, 0.0) for metric in metrics]
        figure.add_trace(
            go.Scatterpolar(
                r=radial_values + [radial_values[0]],
                theta=metrics + [metrics[0]],
                fill="toself",
                name=variant,
            )
        )

    figure.update_layout(
        title="RAGAS Ablation Radar",
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        showlegend=True,
    )
    figure.write_html(str(output_path), include_plotlyjs="cdn")


def _save_bar_chart(results: dict[str, dict[str, float]], output_path: Path) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "recall_at_5"]
    variants = list(results.keys())

    figure = go.Figure()
    for metric in metrics:
        figure.add_trace(
            go.Bar(
                name=metric,
                x=variants,
                y=[results[variant].get(metric, 0.0) for variant in variants],
            )
        )

    figure.update_layout(
        title="RAGAS Ablation Bar Chart",
        barmode="group",
        yaxis_title="Score",
        yaxis_range=[0, 1],
    )
    figure.write_html(str(output_path), include_plotlyjs="cdn")


if __name__ == "__main__":
    run()
# STEP_2_SUMMARY: Replaced custom ablation runner with RAGAS-driven 4-variant evaluation while preserving CLI compatibility.
# STEP_3_SUMMARY: Added summary table and plotly visual outputs for RAGAS ablation metrics.
