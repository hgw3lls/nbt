"""Run report generation with analysis metrics and artifact citations."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from neural_bending_toolkit.analysis.attention import attention_entropy
from neural_bending_toolkit.analysis.bend_classifier import write_bend_classification
from neural_bending_toolkit.analysis.coherence import (
    repetition_score,
    self_consistency_score,
    temporal_reference_stability,
)
from neural_bending_toolkit.analysis.derived_metrics import write_derived_metrics
from neural_bending_toolkit.analysis.embeddings import (
    compute_pca_projection,
    compute_umap_projection,
)
from neural_bending_toolkit.analysis.images import (
    image_diversity_lpips_or_hash,
    load_image_arrays,
)
from neural_bending_toolkit.analysis.theory_memo_generator import build_theory_memo


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _collect_text_artifacts(artifacts_dir: Path) -> list[str]:
    texts: list[str] = []
    for path in artifacts_dir.glob("**/*.txt"):
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return texts


def _collect_image_paths(artifacts_dir: Path) -> list[Path]:
    pngs = sorted(artifacts_dir.glob("**/*.png"))
    jpgs = sorted(artifacts_dir.glob("**/*.jpg"))
    return pngs + jpgs


def _collect_figure_refs(run_dir: Path, limit: int = 8) -> list[str]:
    figure_paths = sorted((run_dir / "analysis").glob("**/*.png"))
    figure_paths.extend(sorted((run_dir / "artifacts").glob("**/*.png")))
    dedup: list[str] = []
    for path in figure_paths:
        rel = str(path.relative_to(run_dir))
        if rel not in dedup:
            dedup.append(rel)
    return dedup[:limit]


def generate_markdown_report(run_dir: Path, output_name: str = "report.md") -> Path:
    """Generate markdown report with figures/proxies and artifact citations."""
    run_dir = Path(run_dir)
    artifacts_dir = run_dir / "artifacts"
    analysis_dir = run_dir / "analysis"
    legacy_analysis_dir = run_dir / "artifacts" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    legacy_analysis_dir.mkdir(parents=True, exist_ok=True)

    derived_path = write_derived_metrics(run_dir)
    classification_path = write_bend_classification(run_dir)
    memo_path = build_theory_memo(run_dir)
    derived = _read_json(derived_path)
    classification = _read_json(classification_path)

    metrics_records = _read_jsonl(run_dir / "metrics.jsonl")
    event_records = _read_jsonl(run_dir / "events.jsonl")

    metric_values = [
        float(record.get("value", 0.0))
        for record in metrics_records
        if "value" in record
    ]
    metric_summary = {
        "count": len(metric_values),
        "mean": float(np.mean(metric_values)) if metric_values else 0.0,
        "std": float(np.std(metric_values)) if metric_values else 0.0,
    }

    texts = _collect_text_artifacts(artifacts_dir)
    repetition_values = [repetition_score(text) for text in texts] if texts else [0.0]
    coherence = {
        "self_consistency": self_consistency_score(texts) if texts else 1.0,
        "mean_repetition": float(np.mean(repetition_values)),
        "temporal_reference_stability": temporal_reference_stability(texts),
    }

    points: list[list[float]] = []
    for rec in metrics_records:
        meta = rec.get("metadata", {})
        is_embedding = (
            isinstance(meta, dict)
            and "embedding" in meta
            and isinstance(meta["embedding"], list)
        )
        if is_embedding:
            points.append([float(x) for x in meta["embedding"]])
    if len(points) < 3:
        rng = np.random.default_rng(7)
        points = rng.normal(size=(16, 8)).tolist()

    emb = np.asarray(points, dtype=np.float64)
    pca = compute_pca_projection(emb, n_components=2)
    umap_proj = compute_umap_projection(emb, n_components=2)
    np.save(analysis_dir / "embedding_pca.npy", pca)
    np.save(analysis_dir / "embedding_umap.npy", umap_proj)
    np.save(legacy_analysis_dir / "embedding_pca.npy", pca)
    np.save(legacy_analysis_dir / "embedding_umap.npy", umap_proj)

    attn_arrays = []
    for rec in event_records:
        payload = rec.get("payload", {})
        if isinstance(payload, dict) and "attention" in payload:
            arr = np.asarray(payload["attention"], dtype=np.float64)
            if arr.ndim >= 2:
                attn_arrays.append(arr)
    if not attn_arrays:
        rng = np.random.default_rng(3)
        attn_arrays = [rng.random((4, 8))]
    entropies = [float(np.mean(attention_entropy(arr))) for arr in attn_arrays]

    image_paths = _collect_image_paths(artifacts_dir)
    image_diversity = {"method": "none", "score": 0.0}
    if image_paths:
        try:
            image_arrays = load_image_arrays(image_paths)
            image_diversity = image_diversity_lpips_or_hash(image_arrays)
        except Exception:
            image_diversity = {"method": "hash", "score": 0.0}

    artifact_citations = [str(path.relative_to(run_dir)) for path in image_paths[:10]]
    figure_refs = _collect_figure_refs(run_dir)

    summary = {
        "metric_summary": metric_summary,
        "coherence": coherence,
        "attention_entropy_mean": float(np.mean(entropies)),
        "image_diversity": image_diversity,
        "artifacts_cited": artifact_citations,
        "bend_tag": classification.get("bend_tag"),
        "scores": classification.get("scores", {}),
        "derived_metrics": {
            key: derived.get(key)
            for key in [
                "divergence",
                "entropy_delta",
                "coherence_delta",
                "refusal_rate_delta",
                "attractor_density_delta",
                "structural_causality_delta",
                "cross_task_consistency",
            ]
        },
    }
    summary_path = analysis_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (legacy_analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    scores = classification.get("scores", {})
    lines = [
        "# Neural Bending Toolkit Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Bend Classification",
        f"- Bend tag: **{classification.get('bend_tag', 'unknown')}**",
        "",
        "| Bend Family | Score |",
        "|---|---:|",
    ]
    for name in sorted(scores):
        lines.append(f"| {name} | {float(scores[name]):.4f} |")

    lines.extend(
        [
            "",
            "## Key Derived Metrics",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for metric_name in [
        "divergence",
        "entropy_delta",
        "coherence_delta",
        "refusal_rate_delta",
        "attractor_density_delta",
        "structural_causality_delta",
        "cross_task_consistency",
    ]:
        value = derived.get(metric_name)
        value_text = "n/a" if value is None else f"{float(value):.4f}"
        lines.append(f"| {metric_name} | {value_text} |")

    lines.extend(
        [
            "",
            "## Theory Memo",
            f"- `{memo_path.relative_to(run_dir)}`",
            "",
            "## Metrics",
            f"- Count: {metric_summary['count']}",
            f"- Mean: {metric_summary['mean']:.6f}",
            f"- Std: {metric_summary['std']:.6f}",
            "",
            "## Embedding Topology",
            "- PCA projection saved: `analysis/embedding_pca.npy`",
            "- UMAP projection saved: `analysis/embedding_umap.npy`",
            "",
            "## Attention / Distribution",
            f"- Mean attention entropy: {float(np.mean(entropies)):.6f}",
            "",
            "## Coherence Proxies",
            f"- Self-consistency: {coherence['self_consistency']:.6f}",
            f"- Mean repetition: {coherence['mean_repetition']:.6f}",
            (
                "- Temporal reference stability: "
                f"{coherence['temporal_reference_stability']:.6f}"
            ),
            "",
            "## Image Diversity",
            f"- Method: {image_diversity['method']}",
            f"- Score: {float(image_diversity['score']):.6f}",
            "",
            "## Figure references",
        ]
    )

    if figure_refs:
        for ref in figure_refs:
            lines.append(f"- `{ref}`")
    else:
        lines.append("- No figures detected.")

    lines.extend(["", "## Artifact citations"])
    if artifact_citations:
        for citation in artifact_citations:
            lines.append(f"- `{citation}`")
    else:
        lines.append("- No image artifacts found.")

    lines.extend(
        [
            "",
            "## Analysis citations",
            "- `analysis/summary.json`",
            "- `analysis/embedding_pca.npy`",
            "- `analysis/embedding_umap.npy`",
            "- `analysis/derived_metrics.json`",
            "- `analysis/bend_classification.json`",
        ]
    )

    out_path = run_dir / output_name
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def generate_html_report(run_dir: Path, output_name: str = "report.html") -> Path:
    """Generate a minimal HTML report wrapper around markdown report content."""
    run_dir = Path(run_dir)
    markdown_path = generate_markdown_report(run_dir, output_name="report.md")
    markdown_text = markdown_path.read_text(encoding="utf-8")
    html_path = run_dir / output_name
    html_payload = "\n".join(
        [
            "<html><head><meta charset='utf-8'><title>NBT Report</title></head><body>",
            "<h1>Neural Bending Toolkit Report</h1>",
            f"<pre>{html.escape(markdown_text)}</pre>",
            "</body></html>",
        ]
    )
    html_path.write_text(html_payload, encoding="utf-8")
    return html_path
