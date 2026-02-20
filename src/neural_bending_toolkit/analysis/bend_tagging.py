"""Tagging and comparison reporting for flagship metastability experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

# Tunable constants for explicit thresholding behavior.
RECOVERY_HIGH = 0.65
RECOVERY_LOW = 0.20
COLLAPSE_HIGH = 0.35
COLLAPSE_MODERATE_LOW = 0.08
COLLAPSE_MODERATE_HIGH = 0.35
BASIN_SMALL = 500.0
BASIN_LARGE = 2500.0
VARIANCE_DEGENERATE = 5.0


class BendTaggingResult(BaseModel):
    """Structured output for bend tagging."""

    tags: list[str] = Field(default_factory=list)
    scores: dict[str, float]
    diagnostics: dict[str, Any] = Field(default_factory=dict)



def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _mean_metric(mapping: dict[str, float] | None) -> float:
    if not isinstance(mapping, dict):
        return 0.0
    vals = [float(v) for v in mapping.values() if isinstance(v, (int, float))]
    return _mean(vals)


def score_and_tag_metastability(
    *,
    recovery_index_value: float,
    concentration_collapse_index_value: float,
    basin_shift_proxy: dict[str, Any] | None,
    output_variance: float | None = None,
) -> BendTaggingResult:
    """Compute simple tags + score breakdown from metastability metrics."""

    basin_mse = _mean_metric((basin_shift_proxy or {}).get("pairwise_mse"))
    is_degenerate = output_variance is not None and output_variance <= VARIANCE_DEGENERATE

    revelatory_score = (
        1.2 * max(0.0, recovery_index_value)
        + 0.6 * max(0.0, COLLAPSE_MODERATE_HIGH - abs(concentration_collapse_index_value - 0.2))
        + 0.4 * max(0.0, BASIN_SMALL - basin_mse) / max(BASIN_SMALL, 1e-8)
    )
    disruptive_score = (
        1.3 * max(0.0, concentration_collapse_index_value - COLLAPSE_HIGH)
        + 1.0 * max(0.0, RECOVERY_LOW - recovery_index_value)
        + 0.8 * max(0.0, basin_mse - BASIN_LARGE) / max(BASIN_LARGE, 1e-8)
        + (0.6 if is_degenerate else 0.0)
    )
    recoherent_score = (
        1.1 * max(0.0, recovery_index_value - RECOVERY_HIGH)
        + 0.6
        * max(
            0.0,
            min(
                concentration_collapse_index_value - COLLAPSE_MODERATE_LOW,
                COLLAPSE_MODERATE_HIGH - concentration_collapse_index_value,
            ),
        )
        + 0.5 * max(0.0, basin_mse - BASIN_SMALL) / max(BASIN_LARGE - BASIN_SMALL, 1e-8)
        + (0.3 if not is_degenerate else -0.3)
    )

    tags: list[str] = []
    if (
        basin_mse <= BASIN_SMALL
        and recovery_index_value >= RECOVERY_HIGH
        and COLLAPSE_MODERATE_LOW <= concentration_collapse_index_value <= COLLAPSE_MODERATE_HIGH
    ):
        tags.append("revelatory")

    if (
        concentration_collapse_index_value >= COLLAPSE_HIGH
        and recovery_index_value <= RECOVERY_LOW
        and (basin_mse >= BASIN_LARGE or is_degenerate)
    ):
        tags.append("disruptive")

    if (
        COLLAPSE_MODERATE_LOW <= concentration_collapse_index_value <= COLLAPSE_MODERATE_HIGH
        and recovery_index_value >= RECOVERY_HIGH
        and basin_mse >= BASIN_SMALL
        and not is_degenerate
    ):
        tags.append("recoherent")

    if not tags:
        ranked = sorted(
            [
                ("revelatory", revelatory_score),
                ("disruptive", disruptive_score),
                ("recoherent", recoherent_score),
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        tags = [ranked[0][0]]

    return BendTaggingResult(
        tags=tags,
        scores={
            "revelatory": float(revelatory_score),
            "disruptive": float(disruptive_score),
            "recoherent": float(recoherent_score),
        },
        diagnostics={
            "recovery_index": float(recovery_index_value),
            "concentration_collapse_index": float(concentration_collapse_index_value),
            "mean_pairwise_basin_mse": float(basin_mse),
            "output_variance": None if output_variance is None else float(output_variance),
            "degenerate_output": bool(is_degenerate),
            "thresholds": {
                "recovery_high": RECOVERY_HIGH,
                "recovery_low": RECOVERY_LOW,
                "collapse_high": COLLAPSE_HIGH,
                "collapse_moderate_low": COLLAPSE_MODERATE_LOW,
                "collapse_moderate_high": COLLAPSE_MODERATE_HIGH,
                "basin_small": BASIN_SMALL,
                "basin_large": BASIN_LARGE,
                "variance_degenerate": VARIANCE_DEGENERATE,
            },
        },
    )


def write_comparison_report(
    run_dir: Path,
    *,
    comparisons: dict[str, Any],
    summary: dict[str, Any],
) -> Path:
    """Write a concise comparison report artifact under run_dir/comparisons/."""

    run_dir = Path(run_dir)
    out_dir = run_dir / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"comparisons": comparisons, "summary": summary}
    out_path = out_dir / "comparison_report.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
