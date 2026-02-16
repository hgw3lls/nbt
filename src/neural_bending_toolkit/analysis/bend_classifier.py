"""Bend family classifier based on derived metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from neural_bending_toolkit.analysis.derived_metrics import (
    DerivedMetrics,
    compute_derived_metrics,
)


class BendClassification(BaseModel):
    """Standard bend classification payload."""

    bend_tag: str
    scores: dict[str, float]
    gates_passed: dict[str, bool]
    justification: str
    timestamp: str
    geopolitical_flag: bool = False


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_derived(run_dir: Path) -> DerivedMetrics:
    path = run_dir / "analysis" / "derived_metrics.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DerivedMetrics.model_validate(payload)
    return compute_derived_metrics(run_dir)


def _load_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _nz(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def _score_revelatory(metrics: DerivedMetrics) -> float:
    return (
        0.35 * _nz(metrics.structural_causality_delta)
        + 0.25 * _nz(metrics.attractor_density_delta)
        + 0.20 * _nz(metrics.cross_task_consistency)
        + 0.10 * _nz(metrics.coherence_delta)
        - 0.10 * max(0.0, _nz(metrics.refusal_rate_delta))
    )


def _score_disruptive(metrics: DerivedMetrics) -> float:
    return (
        0.30 * abs(_nz(metrics.divergence))
        + 0.25 * max(0.0, _nz(metrics.entropy_delta))
        + 0.20 * max(0.0, -_nz(metrics.coherence_delta))
        + 0.20 * max(0.0, _nz(metrics.refusal_rate_delta))
        - 0.05 * max(0.0, _nz(metrics.structural_causality_delta))
    )


def _score_recoherent(metrics: DerivedMetrics) -> float:
    return (
        0.35 * max(0.0, _nz(metrics.coherence_delta))
        + 0.20 * max(0.0, -_nz(metrics.entropy_delta))
        + 0.20 * max(0.0, -_nz(metrics.refusal_rate_delta))
        + 0.15 * max(0.0, _nz(metrics.structural_causality_delta))
        + 0.10 * max(0.0, _nz(metrics.cross_task_consistency))
        - 0.10 * abs(_nz(metrics.divergence))
    )


def _gates(metrics: DerivedMetrics) -> dict[str, bool]:
    return {
        "revelatory": _nz(metrics.structural_causality_delta) > 0.1
        and _nz(metrics.attractor_density_delta) > 0.0,
        "disruptive": abs(_nz(metrics.divergence)) >= 0.2
        and _nz(metrics.coherence_delta) < 0.0,
        "recoherent": _nz(metrics.coherence_delta) >= 0.1
        and _nz(metrics.refusal_rate_delta) <= 0.1,
    }


def _is_geopolitical_comparative(run_dir: Path) -> bool:
    config = _load_config(run_dir)

    geopolitical = config.get("geopolitical") if isinstance(config, dict) else None
    if isinstance(geopolitical, dict):
        models = geopolitical.get("models")
        prompt_pairs = geopolitical.get("prompt_pairs")
        if isinstance(models, list) and len(models) > 1 and prompt_pairs:
            return True

    models = config.get("models") if isinstance(config, dict) else None
    if isinstance(models, list) and len(models) > 1:
        return True

    comparisons = config.get("comparison_runs") if isinstance(config, dict) else None
    return isinstance(comparisons, list) and len(comparisons) >= 2


def classify_bend(run_dir: Path, epsilon: float = 0.075) -> BendClassification:
    """Classify bend family from derived metrics and metadata."""
    run_dir = Path(run_dir)
    metrics = _load_derived(run_dir)
    scores = {
        "revelatory": _score_revelatory(metrics),
        "disruptive": _score_disruptive(metrics),
        "recoherent": _score_recoherent(metrics),
    }
    gates = _gates(metrics)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_tag = ranked[0][0]

    passed_tags = [tag for tag, passed in gates.items() if passed]
    if passed_tags:
        passed_ranked = sorted(
            [(tag, scores[tag]) for tag in passed_tags],
            key=lambda item: item[1],
            reverse=True,
        )
        best_tag = passed_ranked[0][0]

    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) <= epsilon:
        best_tag = f"hybrid:{ranked[0][0]}+{ranked[1][0]}"

    geopolitical_flag = _is_geopolitical_comparative(run_dir)

    justification = (
        f"Selected '{best_tag}' with scores {scores} and gates {gates}. "
        f"Geopolitical comparative metadata detected={geopolitical_flag}."
    )

    return BendClassification(
        bend_tag=best_tag,
        scores={key: float(value) for key, value in scores.items()},
        gates_passed=gates,
        justification=justification,
        timestamp=_timestamp(),
        geopolitical_flag=geopolitical_flag,
    )


def write_bend_classification(run_dir: Path) -> Path:
    """Persist bend classification JSON in run_dir/analysis/."""
    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    payload = classify_bend(run_dir)
    out_path = analysis_dir / "bend_classification.json"
    out_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    return out_path
