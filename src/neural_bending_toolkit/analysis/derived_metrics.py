"""Compute standardized derived metrics from run artifacts and logs."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
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
    "attention_entropy_mean_per_step",
    "attention_entropy_variance_per_step",
    "attention_concentration_topk_mass",
    "attention_entropy_recovery_index",
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
    attention_entropy_mean_per_step: float | None = None
    attention_entropy_variance_per_step: float | None = None
    attention_concentration_topk_mass: float | None = None
    attention_entropy_recovery_index: float | None = None
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


def _metric_rows_by_step(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
) -> dict[int, list[float]]:
    by_step: dict[int, list[float]] = defaultdict(list)
    lowered_name = metric_name.lower()
    for row in rows:
        if str(row.get("metric_name", "")).strip().lower() != lowered_name:
            continue
        step = row.get("step")
        value = row.get("value")
        if not isinstance(step, int) or not isinstance(value, (int, float)):
            continue
        by_step[step].append(float(value))
    return dict(by_step)


def _mean_per_step(by_step: dict[int, list[float]]) -> dict[int, float]:
    return {step: float(np.mean(values)) for step, values in by_step.items() if values}


def _variance_per_step(by_step: dict[int, list[float]]) -> dict[int, float]:
    return {step: float(np.var(values)) for step, values in by_step.items() if values}


def _recovery_index(entropy_step_means: dict[int, float]) -> float | None:
    if len(entropy_step_means) < 2:
        return None
    ordered_steps = sorted(entropy_step_means)
    values = [entropy_step_means[step] for step in ordered_steps]
    third = max(1, len(values) // 3)
    early = float(np.mean(values[:third]))
    late = float(np.mean(values[-third:]))
    return float(abs(late - early))


def _topk_mass_from_heatmaps(run_dir: Path, k: int = 5) -> float | None:
    files = sorted((run_dir / "artifacts").glob("*_step_*.npy"))
    masses: list[float] = []
    for path in files:
        try:
            arr = np.load(path)
        except Exception:
            continue
        if arr.ndim < 1 or arr.size == 0:
            continue
        probs = np.asarray(arr, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs[None, :]
        probs = probs.reshape(-1, probs.shape[-1])
        row_sums = probs.sum(axis=1, keepdims=True)
        valid = row_sums.squeeze(-1) > 0
        if not np.any(valid):
            continue
        normed = probs[valid] / row_sums[valid]
        kk = min(k, normed.shape[-1])
        top = np.partition(normed, normed.shape[-1] - kk, axis=1)[:, -kk:]
        masses.append(float(np.mean(top.sum(axis=1))))
    if not masses:
        return None
    return float(np.mean(masses))


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

    latest_aliases = {
        key[:-7]: value for key, value in source.items() if key.endswith(".latest")
    }
    source.update(latest_aliases)
    return source


def compute_derived_metrics(run_dir: Path) -> DerivedMetrics:
    """Compute standardized derived metrics for a run directory."""
    run_dir = Path(run_dir)
    source = _build_numeric_source(run_dir)
    metric_rows = _read_jsonl(run_dir / "metrics.jsonl")

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

    entropy_step_means = _mean_per_step(
        _metric_rows_by_step(metric_rows, metric_name="attention_entropy")
    )
    entropy_step_vars = _variance_per_step(
        _metric_rows_by_step(metric_rows, metric_name="attention_entropy")
    )
    topk_by_step = _mean_per_step(
        _metric_rows_by_step(metric_rows, metric_name="attention_topk_mass")
    )

    attention_entropy_mean_per_step = (
        float(np.mean(list(entropy_step_means.values())))
        if entropy_step_means
        else None
    )
    attention_entropy_variance_per_step = (
        float(np.mean(list(entropy_step_vars.values())))
        if entropy_step_vars
        else None
    )
    attention_concentration_topk_mass = (
        float(np.mean(list(topk_by_step.values())))
        if topk_by_step
        else _topk_mass_from_heatmaps(run_dir)
    )
    attention_entropy_recovery_index = _recovery_index(entropy_step_means)

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
        attention_entropy_mean_per_step=attention_entropy_mean_per_step,
        attention_entropy_variance_per_step=attention_entropy_variance_per_step,
        attention_concentration_topk_mass=attention_concentration_topk_mass,
        attention_entropy_recovery_index=attention_entropy_recovery_index,
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
            attention_entropy_mean_per_step,
            attention_entropy_variance_per_step,
            attention_concentration_topk_mass,
            attention_entropy_recovery_index,
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


def _append_trace_metric_records(run_dir: Path, derived: DerivedMetrics) -> None:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return

    existing = _read_jsonl(metrics_path)
    existing_names = {
        str(row.get("metric_name", "")).strip().lower() for row in existing
    }

    payloads = [
        (
            "attention_entropy_mean_per_step",
            derived.attention_entropy_mean_per_step,
        ),
        (
            "attention_entropy_variance_per_step",
            derived.attention_entropy_variance_per_step,
        ),
        (
            "attention_concentration_topk_mass",
            derived.attention_concentration_topk_mass,
        ),
        (
            "attention_entropy_recovery_index",
            derived.attention_entropy_recovery_index,
        ),
    ]

    lines: list[str] = []
    for metric_name, value in payloads:
        if value is None or metric_name.lower() in existing_names:
            continue
        record = {
            "timestamp": _timestamp(),
            "step": -1,
            "metric_name": metric_name,
            "value": value,
            "metadata": {"source": "derived_metrics"},
        }
        lines.append(json.dumps(record))

    if lines:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


def _write_run_summary(run_dir: Path, derived: DerivedMetrics) -> None:
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    summary["bend_v2_trace_metrics"] = {
        "attention_entropy_mean_per_step": derived.attention_entropy_mean_per_step,
        "attention_entropy_variance_per_step": (
            derived.attention_entropy_variance_per_step
        ),
        "attention_concentration_topk_mass": derived.attention_concentration_topk_mass,
        "attention_entropy_recovery_index": derived.attention_entropy_recovery_index,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_derived_metrics(run_dir: Path) -> Path:
    """Compute and persist derived metrics JSON in run_dir/analysis/."""
    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    derived = compute_derived_metrics(run_dir)
    out_path = analysis_dir / "derived_metrics.json"
    out_path.write_text(derived.model_dump_json(indent=2), encoding="utf-8")

    _append_trace_metric_records(run_dir, derived)
    _write_run_summary(run_dir, derived)
    return out_path
