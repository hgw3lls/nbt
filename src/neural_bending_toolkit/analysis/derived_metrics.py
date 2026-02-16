"""Compute standardized derived metrics from run artifacts and logs."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import yaml
from pydantic import BaseModel, Field

DERIVED_METRIC_KEYS = [
    "divergence",
    "entropy_delta",
    "coherence_delta",
    "refusal_rate_delta",
    "attractor_density_delta",
    "structural_causality_delta",
    "cross_task_consistency",
]


class DerivedMetrics(BaseModel):
    """Standardized run-level derived metrics object."""

    run_dir: str
    generated_at: str
    divergence: float | None = None
    entropy_delta: float | None = None
    coherence_delta: float | None = None
    refusal_rate_delta: float | None = None
    attractor_density_delta: float | None = None
    structural_causality_delta: float | None = None
    cross_task_consistency: float | None = None
    availability: dict[str, bool] = Field(default_factory=dict)
    normalized: dict[str, float | None] = Field(default_factory=dict)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _flatten_values(data: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_values(value, next_prefix))
    elif isinstance(data, list):
        return out
    elif isinstance(data, (int, float)):
        out[prefix.lower()] = float(data)
    return out


def _aggregate_metric_stream(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_name: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        name = str(row.get("metric_name", "")).strip().lower()
        value = row.get("value")
        if not name or not isinstance(value, (int, float)):
            continue
        by_name[name].append(float(value))

    aggregate: dict[str, float] = {}
    for key, values in by_name.items():
        aggregate[key] = float(sum(values) / len(values))
        aggregate[f"{key}.latest"] = float(values[-1])
    return aggregate


def robust_median_iqr(values: list[float]) -> tuple[float, float]:
    """Return robust center/scale estimate using median and IQR."""
    if not values:
        return 0.0, 1.0
    sorted_vals = sorted(values)
    med = float(median(sorted_vals))
    q1 = sorted_vals[len(sorted_vals) // 4]
    q3 = sorted_vals[(3 * len(sorted_vals)) // 4]
    iqr = float(q3 - q1)
    if iqr == 0.0:
        iqr = 1.0
    return med, iqr


def robust_scale(value: float | None, population: list[float]) -> float | None:
    """Scale a value with robust median/IQR normalization."""
    if value is None:
        return None
    med, iqr = robust_median_iqr(population)
    return float((value - med) / iqr)


def _first_numeric(source: dict[str, float], aliases: list[str]) -> float | None:
    for alias in aliases:
        if alias in source:
            return float(source[alias])
    return None


def _delta_from_aliases(
    source: dict[str, float],
    *,
    delta_aliases: list[str],
    current_aliases: list[str],
    baseline_aliases: list[str],
) -> float | None:
    direct = _first_numeric(source, delta_aliases)
    if direct is not None:
        return direct

    current = _first_numeric(source, current_aliases)
    baseline = _first_numeric(source, baseline_aliases)
    if current is None or baseline is None:
        return None
    return float(current - baseline)


def _build_numeric_source(run_dir: Path) -> dict[str, float]:
    metrics = _aggregate_metric_stream(_read_jsonl(run_dir / "metrics.jsonl"))

    summary_candidates = [
        run_dir / "summary.json",
        run_dir / "analysis" / "summary.json",
        run_dir / "artifacts" / "analysis" / "summary.json",
    ]
    summary: dict[str, float] = {}
    for summary_path in summary_candidates:
        summary.update(_flatten_values(_read_json(summary_path)))

    config = _flatten_values(_read_yaml(run_dir / "config.yaml"))

    source = {**config, **summary, **metrics}

    # Add suffix-stripped aliases for "foo.latest" keys.
    latest_aliases = {
        key[:-7]: value for key, value in source.items() if key.endswith(".latest")
    }
    source.update(latest_aliases)
    return source


def compute_derived_metrics(run_dir: Path) -> DerivedMetrics:
    """Compute standardized derived metrics for a run directory."""
    run_dir = Path(run_dir)
    source = _build_numeric_source(run_dir)

    divergence = _first_numeric(
        source,
        [
            "kl_vs_baseline",
            "js_vs_baseline",
            "js_divergence_vs_baseline",
            "divergence",
            "distribution_kl",
        ],
    )

    entropy_delta = _delta_from_aliases(
        source,
        delta_aliases=["entropy_delta", "attention_entropy_delta"],
        current_aliases=["entropy", "attention_entropy_mean"],
        baseline_aliases=["baseline_entropy", "baseline_attention_entropy"],
    )

    repetition_delta = _delta_from_aliases(
        source,
        delta_aliases=["repetition_ratio_delta", "mean_repetition_delta"],
        current_aliases=["repetition_ratio", "coherence.mean_repetition"],
        baseline_aliases=["baseline_repetition_ratio", "baseline_mean_repetition"],
    )
    self_consistency_delta = _delta_from_aliases(
        source,
        delta_aliases=["self_consistency_delta", "coherence_delta_self_consistency"],
        current_aliases=["self_consistency", "coherence.self_consistency"],
        baseline_aliases=["baseline_self_consistency"],
    )
    perplexity_delta = _delta_from_aliases(
        source,
        delta_aliases=["perplexity_delta"],
        current_aliases=["perplexity"],
        baseline_aliases=["baseline_perplexity"],
    )
    direct_coherence_delta = _first_numeric(source, ["coherence_delta"])

    coherence_parts: list[float] = []
    if direct_coherence_delta is not None:
        coherence_parts.append(direct_coherence_delta)
    if repetition_delta is not None:
        coherence_parts.append(-repetition_delta)
    if self_consistency_delta is not None:
        coherence_parts.append(self_consistency_delta)
    if perplexity_delta is not None:
        coherence_parts.append(-perplexity_delta)
    coherence_delta = (
        float(sum(coherence_parts) / len(coherence_parts)) if coherence_parts else None
    )

    refusal_rate_delta = _delta_from_aliases(
        source,
        delta_aliases=["refusal_rate_delta", "delta_refusal_rate"],
        current_aliases=["refusal_rate", "summary.refusal_rate"],
        baseline_aliases=["baseline_refusal_rate"],
    )

    attractor_density_delta = _delta_from_aliases(
        source,
        delta_aliases=["attractor_density_delta", "density_change"],
        current_aliases=["attractor_density", "bent_density"],
        baseline_aliases=["baseline_attractor_density", "baseline_density"],
    )

    structural_causality_delta = _delta_from_aliases(
        source,
        delta_aliases=["structural_causality_delta", "causality_delta"],
        current_aliases=["structural_causality", "structural_causality_score"],
        baseline_aliases=["baseline_structural_causality"],
    )

    cross_task_consistency = _first_numeric(
        source,
        [
            "cross_task_consistency",
            "task_consistency",
            "cross_task_alignment",
        ],
    )

    metrics_payload = DerivedMetrics(
        run_dir=str(run_dir),
        generated_at=_timestamp(),
        divergence=divergence,
        entropy_delta=entropy_delta,
        coherence_delta=coherence_delta,
        refusal_rate_delta=refusal_rate_delta,
        attractor_density_delta=attractor_density_delta,
        structural_causality_delta=structural_causality_delta,
        cross_task_consistency=cross_task_consistency,
    )

    observed_values = [
        float(value)
        for value in [
            divergence,
            entropy_delta,
            coherence_delta,
            refusal_rate_delta,
            attractor_density_delta,
            structural_causality_delta,
            cross_task_consistency,
        ]
        if value is not None
    ]

    metrics_payload.availability = {
        key: getattr(metrics_payload, key) is not None for key in DERIVED_METRIC_KEYS
    }
    metrics_payload.normalized = {
        key: robust_scale(getattr(metrics_payload, key), observed_values)
        for key in DERIVED_METRIC_KEYS
    }
    return metrics_payload


def write_derived_metrics(run_dir: Path) -> Path:
    """Compute and persist derived metrics JSON in run_dir/analysis/."""
    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    derived = compute_derived_metrics(run_dir)
    out_path = analysis_dir / "derived_metrics.json"
    out_path.write_text(derived.model_dump_json(indent=2), encoding="utf-8")
    return out_path
