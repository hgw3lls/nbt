"""Norm-focused metastability metrics and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def variance_profile(
    trace_rows: list[dict[str, Any]],
    *,
    condition: str,
) -> dict[int, float]:
    """Aggregate per-step norm_output_var values for one condition."""

    buckets: dict[int, list[float]] = {}
    for row in trace_rows:
        if row.get("metric_name") != "norm_output_var":
            continue
        metadata = row.get("metadata", {})
        if metadata.get("condition") != condition:
            continue
        step = int(row.get("step", 0))
        value = float(row.get("value", float("nan")))
        buckets.setdefault(step, []).append(value)

    profile: dict[int, float] = {}
    for step, values in sorted(buckets.items()):
        if not values:
            continue
        profile[step] = float(np.mean(values))
    return profile


def collapse_index(
    profile: dict[int, float],
    *,
    pre_window: tuple[int, int],
    during_window: tuple[int, int],
    eps: float = 1e-8,
) -> float:
    """Measure variance drop from pre-window to shock window."""

    pre_vals = [v for s, v in profile.items() if pre_window[0] <= s <= pre_window[1]]
    during_vals = [
        v for s, v in profile.items() if during_window[0] <= s <= during_window[1]
    ]
    if not pre_vals or not during_vals:
        return 0.0

    pre_mean = float(np.nanmean(pre_vals))
    during_mean = float(np.nanmean(during_vals))
    if not np.isfinite(pre_mean) or not np.isfinite(during_mean):
        return 0.0

    drop = max(pre_mean - during_mean, 0.0)
    return float(drop / (abs(pre_mean) + eps))


def recovery_index(
    profile: dict[int, float],
    *,
    during_window: tuple[int, int],
    post_window: tuple[int, int],
    eps: float = 1e-8,
) -> float:
    """Measure normalized rebound from shock window to post window."""

    during_vals = [
        v for s, v in profile.items() if during_window[0] <= s <= during_window[1]
    ]
    post_vals = [v for s, v in profile.items() if post_window[0] <= s <= post_window[1]]
    if not during_vals or not post_vals:
        return 0.0

    during_mean = float(np.nanmean(during_vals))
    post_mean = float(np.nanmean(post_vals))
    if not np.isfinite(during_mean) or not np.isfinite(post_mean):
        return 0.0

    rebound = max(post_mean - during_mean, 0.0)
    return float(rebound / (abs(during_mean) + eps))


def stability_proxy(profile: dict[int, float]) -> float:
    """Fraction of steps with finite norm variance stats."""

    if not profile:
        return 0.0
    values = np.asarray(list(profile.values()), dtype=np.float64)
    return float(np.isfinite(values).mean())


def plot_norm_variance_over_steps(
    variance_profiles: dict[str, dict[int, float]],
    output_path: Path,
) -> Path:
    """Plot variance trajectories for baseline/collapse/recovery conditions."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 4.5))
    ordered = ["baseline", "norm_collapse", "norm_recovery"]
    for condition in ordered:
        profile = variance_profiles.get(condition, {})
        if not profile:
            continue
        steps = sorted(profile.keys())
        values = [profile[s] for s in steps]
        plt.plot(steps, values, label=condition)

    plt.xlabel("Step")
    plt.ylabel("Norm output variance")
    plt.title("Norm variance over denoising steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
