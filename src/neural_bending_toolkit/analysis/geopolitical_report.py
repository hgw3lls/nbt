"""Geopolitical report generation and multi-run comparison utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunGeopoliticalSummary:
    """Aggregated metrics extracted from one geopolitical run."""

    run_dir: Path
    refusal_rate: float
    average_sentiment_delta: float
    average_attractor_density_change: float
    prompt_pair_count: int
    token_set_count: int


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ensure_reports_dir(reports_dir: Path | None = None) -> Path:
    out = reports_dir or Path("reports")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_phase2_rows(run_dir: Path) -> list[dict[str, Any]]:
    json_path = (
        run_dir
        / "artifacts"
        / "geopolitical"
        / "phase_2_governance_dissonance"
        / "governance_dissonance_results.json"
    )
    loaded = _load_json(json_path)
    if isinstance(loaded, list):
        return loaded
    return []


def _load_phase3_rows(run_dir: Path) -> list[dict[str, Any]]:
    json_path = (
        run_dir
        / "artifacts"
        / "geopolitical"
        / "phase_3_justice_attractors"
        / "justice_attractor_results.json"
    )
    loaded = _load_json(json_path)
    if isinstance(loaded, list):
        return loaded
    return []


def _plot_similarity_heatmaps(phase1_dir: Path, figs_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for csv_path in sorted(phase1_dir.glob("*_similarity.csv")):
        rows: list[list[float]] = []
        labels: list[str] = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                labels.append(row[0])
                rows.append([float(value) for value in row[1:]])

        if not rows:
            continue

        matrix = np.asarray(rows, dtype=np.float64)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
        ax.set_title(f"Embedding Similarity: {csv_path.stem}")
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        out_path = figs_dir / f"{csv_path.stem}_heatmap.png"
        fig.savefig(out_path)
        plt.close(fig)
        outputs.append(out_path)

    return outputs


def _plot_projection(
    figs_dir: Path, source_path: Path, title: str, suffix: str
) -> Path | None:
    if not source_path.exists():
        return None

    projection = np.load(source_path)
    if projection.ndim != 2 or projection.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(projection[:, 0], projection[:, 1], alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()

    out_path = figs_dir / f"{source_path.stem}_{suffix}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_sentiment_comparison(
    rows: list[dict[str, Any]], figs_dir: Path
) -> Path | None:
    if not rows:
        return None

    sentiment_a = [float(row.get("sentiment_a", 0.0)) for row in rows]
    sentiment_b = [float(row.get("sentiment_b", 0.0)) for row in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    ax.bar(x - width / 2, sentiment_a, width, label="Prompt A")
    ax.bar(x + width / 2, sentiment_b, width, label="Prompt B")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Sentiment Comparison by Prompt Pair")
    ax.set_xlabel("Prompt Pair Index")
    ax.set_ylabel("Sentiment Score")
    ax.set_xticks(x, [str(idx + 1) for idx in x])
    ax.legend()
    fig.tight_layout()

    out_path = figs_dir / "sentiment_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_attractor_density(rows: list[dict[str, Any]], figs_dir: Path) -> Path | None:
    if not rows:
        return None

    baseline = [float(row.get("baseline_density", 0.0)) for row in rows]
    bent = [float(row.get("bent_density", 0.0)) for row in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    ax.bar(x - width / 2, baseline, width, label="Baseline")
    ax.bar(x + width / 2, bent, width, label="Bent")
    ax.set_title("Attractor Token Density Comparison")
    ax.set_xlabel("Token Set Index")
    ax.set_ylabel("Density")
    ax.set_xticks(x, [str(idx + 1) for idx in x])
    ax.legend()
    fig.tight_layout()

    out_path = figs_dir / "attractor_density_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def summarize_geopolitical_run(run_dir: Path) -> RunGeopoliticalSummary:
    """Build summary statistics from a geopolitical run directory."""
    phase2_rows = _load_phase2_rows(run_dir)
    phase3_rows = _load_phase3_rows(run_dir)

    refusal_flags = []
    sentiment_deltas = []
    for row in phase2_rows:
        refusal_flags.extend(
            [bool(row.get("refusal_a", False)), bool(row.get("refusal_b", False))]
        )
        sentiment_deltas.append(
            float(row.get("sentiment_b", 0.0)) - float(row.get("sentiment_a", 0.0))
        )

    refusal_rate = float(np.mean(refusal_flags)) if refusal_flags else 0.0
    avg_sentiment_delta = float(np.mean(sentiment_deltas)) if sentiment_deltas else 0.0

    density_changes = [float(row.get("density_change", 0.0)) for row in phase3_rows]
    avg_density_change = float(np.mean(density_changes)) if density_changes else 0.0

    return RunGeopoliticalSummary(
        run_dir=run_dir,
        refusal_rate=refusal_rate,
        average_sentiment_delta=avg_sentiment_delta,
        average_attractor_density_change=avg_density_change,
        prompt_pair_count=len(phase2_rows),
        token_set_count=len(phase3_rows),
    )


def generate_geopolitical_report(
    run_dir: Path,
    *,
    format: str = "markdown",
    reports_dir: Path | None = None,
) -> Path:
    """Generate a geopolitical report (markdown/html) from one run directory."""
    run_dir = Path(run_dir)
    if format not in {"markdown", "html"}:
        raise ValueError("format must be 'markdown' or 'html'")

    reports_root = _ensure_reports_dir(reports_dir)
    report_id = f"geopolitical_report_{_timestamp()}"
    report_dir = reports_root / report_id
    report_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = report_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    phase1_dir = run_dir / "artifacts" / "geopolitical" / "phase_1_ontology_mapping"
    phase2_rows = _load_phase2_rows(run_dir)
    phase3_rows = _load_phase3_rows(run_dir)
    summary = summarize_geopolitical_run(run_dir)

    figure_paths: list[Path] = []
    if phase1_dir.exists():
        figure_paths.extend(_plot_similarity_heatmaps(phase1_dir, figs_dir))

        for npy_path in sorted(phase1_dir.glob("*_pca.npy")):
            plotted = _plot_projection(
                figs_dir,
                npy_path,
                title=f"PCA Projection: {npy_path.stem}",
                suffix="plot",
            )
            if plotted is not None:
                figure_paths.append(plotted)

        for npy_path in sorted(phase1_dir.glob("*_umap.npy")):
            plotted = _plot_projection(
                figs_dir,
                npy_path,
                title=f"UMAP Projection: {npy_path.stem}",
                suffix="plot",
            )
            if plotted is not None:
                figure_paths.append(plotted)

    sentiment_fig = _plot_sentiment_comparison(phase2_rows, figs_dir)
    if sentiment_fig is not None:
        figure_paths.append(sentiment_fig)

    density_fig = _plot_attractor_density(phase3_rows, figs_dir)
    if density_fig is not None:
        figure_paths.append(density_fig)

    refusal_table_lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| Prompt pair count | {summary.prompt_pair_count} |",
        f"| Refusal rate | {summary.refusal_rate:.4f} |",
    ]

    figure_rel = [path.relative_to(report_dir) for path in figure_paths]

    if format == "markdown":
        report_path = report_dir / "report.md"
        lines = [
            "# Geopolitical Bend Report",
            "",
            "## Run Metadata",
            f"- Run directory: `{run_dir}`",
            f"- Generated at (UTC): `{datetime.now(timezone.utc).isoformat()}`",
            "",
            "## Embedding Similarity Heatmaps",
        ]
        if figure_rel:
            heatmaps = [p for p in figure_rel if "heatmap" in p.name]
            lines.extend([f"- `{p}`" for p in heatmaps] or ["- No heatmaps available."])
        else:
            lines.append("- No heatmaps available.")

        lines.extend(
            [
                "",
                "## PCA/UMAP Plots",
                *(
                    [
                        f"- `{p}`"
                        for p in figure_rel
                        if "_pca" in p.name or "_umap" in p.name
                    ]
                    or ["- No PCA/UMAP plots available."]
                ),
                "",
                "## Sentiment Comparison",
                (
                    f"- Figure: `{sentiment_fig.relative_to(report_dir)}`"
                    if sentiment_fig is not None
                    else "- No sentiment figure generated."
                ),
                "",
                "## Refusal Rate Table",
                *refusal_table_lines,
                "",
                "## Attractor Density Comparison",
                (
                    f"- Figure: `{density_fig.relative_to(report_dir)}`"
                    if density_fig is not None
                    else "- No attractor density figure generated."
                ),
                "",
                "## Summary Metrics",
                (
                    "- Average sentiment delta (B - A): "
                    f"`{summary.average_sentiment_delta:.4f}`"
                ),
                (
                    "- Average attractor density change (bent - baseline): "
                    f"`{summary.average_attractor_density_change:.4f}`"
                ),
                f"- Token set count: `{summary.token_set_count}`",
            ]
        )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    report_path = report_dir / "report.html"
    img_tags = "\n".join(
        [
            (
                f"<li><img src='{p.as_posix()}' alt='{p.name}' "
                "style='max-width: 900px;'></li>"
            )
            for p in figure_rel
        ]
    )
    html = f"""<!doctype html>
<html lang='en'>
<head><meta charset='utf-8'><title>Geopolitical Bend Report</title></head>
<body>
<h1>Geopolitical Bend Report</h1>
<h2>Run Metadata</h2>
<p>Run directory: <code>{run_dir}</code></p>
<h2>Embedding Similarity Heatmaps</h2>
<ul>{img_tags}</ul>
<h2>PCA/UMAP Plots</h2>
<ul>{img_tags}</ul>
<h2>Sentiment Comparison Bar Charts</h2>
<ul>{img_tags}</ul>
<h2>Refusal Rate Table</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Prompt pair count</td><td>{summary.prompt_pair_count}</td></tr>
<tr><td>Refusal rate</td><td>{summary.refusal_rate:.4f}</td></tr>
</table>
<h2>Attractor Density Comparison</h2>
<p>Average attractor density change: {summary.average_attractor_density_change:.4f}</p>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def compare_geopolitical_runs(
    run_dirs: list[Path],
    *,
    reports_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Create side-by-side run comparison CSV and markdown summary."""
    if len(run_dirs) < 2:
        raise ValueError("At least two run directories are required for comparison.")

    resolved_runs = [Path(path) for path in run_dirs]
    summaries = [summarize_geopolitical_run(path) for path in resolved_runs]

    reports_root = _ensure_reports_dir(reports_dir)
    compare_id = f"geopolitical_compare_{_timestamp()}"
    compare_dir = reports_root / compare_id
    compare_dir.mkdir(parents=True, exist_ok=True)

    csv_path = compare_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_dir",
                "refusal_rate",
                "average_sentiment_delta",
                "average_attractor_density_change",
                "prompt_pair_count",
                "token_set_count",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "run_dir": str(summary.run_dir),
                    "refusal_rate": f"{summary.refusal_rate:.6f}",
                    "average_sentiment_delta": (
                        f"{summary.average_sentiment_delta:.6f}"
                    ),
                    "average_attractor_density_change": (
                        f"{summary.average_attractor_density_change:.6f}"
                    ),
                    "prompt_pair_count": summary.prompt_pair_count,
                    "token_set_count": summary.token_set_count,
                }
            )

    markdown_path = compare_dir / "comparison.md"
    lines = [
        "# Geopolitical Run Comparison",
        "",
        "## Runs",
        *[f"- `{run}`" for run in resolved_runs],
        "",
        "## Side-by-side metrics",
        (
            "| Run | Refusal Rate | Avg Sentiment Delta (B-A) | "
            "Avg Attractor Density Change | Prompt Pairs | Token Sets |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| "
            f"`{summary.run_dir}` | {summary.refusal_rate:.4f} | "
            f"{summary.average_sentiment_delta:.4f} | "
            f"{summary.average_attractor_density_change:.4f} | "
            f"{summary.prompt_pair_count} | {summary.token_set_count} |"
        )

    lines.extend(
        [
            "",
            "## Outputs",
            f"- CSV: `{csv_path}`",
            f"- Markdown: `{markdown_path}`",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return csv_path, markdown_path
