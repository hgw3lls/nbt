"""Norm-metastability-specific tagging rules."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

NORM_COLLAPSE_HIGH = 0.35
NORM_COLLAPSE_MODERATE_LOW = 0.12
NORM_COLLAPSE_MODERATE_HIGH = 0.35
NORM_RECOVERY_HIGH = 0.65
NORM_RECOVERY_LOW = 0.20
STABILITY_MIN = 1.0
BASIN_SMALL = 500.0


class NormBendTaggingResult(BaseModel):
    """Tagging payload for norm metastability experiments."""

    tags: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)



def score_and_tag_norm_metastability(
    *,
    collapse_index_value: float,
    recovery_index_value: float,
    stability_proxy_value: float,
    final_basin_shift: float,
) -> NormBendTaggingResult:
    """Apply explicit rule-based tags for norm collapse/recovery behavior."""

    tags: list[str] = []

    if (
        stability_proxy_value < STABILITY_MIN
        or (
            collapse_index_value >= NORM_COLLAPSE_HIGH
            and recovery_index_value <= NORM_RECOVERY_LOW
        )
    ):
        tags.append("disruptive")

    if (
        NORM_COLLAPSE_MODERATE_LOW <= collapse_index_value <= NORM_COLLAPSE_MODERATE_HIGH
        and recovery_index_value >= NORM_RECOVERY_HIGH
        and stability_proxy_value >= STABILITY_MIN
    ):
        tags.append("recoherent")

    if (
        collapse_index_value >= NORM_COLLAPSE_MODERATE_LOW
        and final_basin_shift <= BASIN_SMALL
        and stability_proxy_value >= STABILITY_MIN
    ):
        tags.append("revelatory")

    if not tags:
        if recovery_index_value > collapse_index_value:
            tags = ["recoherent"]
        elif stability_proxy_value < STABILITY_MIN:
            tags = ["disruptive"]
        else:
            tags = ["revelatory"]

    return NormBendTaggingResult(
        tags=tags,
        diagnostics={
            "collapse_index": float(collapse_index_value),
            "recovery_index": float(recovery_index_value),
            "stability_proxy": float(stability_proxy_value),
            "final_basin_shift": float(final_basin_shift),
            "thresholds": {
                "collapse_high": NORM_COLLAPSE_HIGH,
                "collapse_moderate_low": NORM_COLLAPSE_MODERATE_LOW,
                "collapse_moderate_high": NORM_COLLAPSE_MODERATE_HIGH,
                "recovery_high": NORM_RECOVERY_HIGH,
                "recovery_low": NORM_RECOVERY_LOW,
                "stability_min": STABILITY_MIN,
                "basin_small": BASIN_SMALL,
            },
        },
    )
