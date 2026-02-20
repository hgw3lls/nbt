"""Residual-stream metastability metrics and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _profile_from_metric(
    trace_rows: list[dict[str, Any]],
    *,
    condition: str,
    metric_name: str,
) -> dict[int, float]:
    buckets: dict[int, list[float]] = {}
    for row in trace_rows:
        if row.get("metric_name") != metric_name:
            continue
        metadata = row.get("metadata", {})
        if metadata.get("condition") != condition:
            continue
        step = int(row.get("step", 0))
        value = float(row.get("value", float("nan")))
        buckets.setdefault(step, []).append(value)

    profile: dict[int, float] = {}
    for step, values in sorted(buckets.items()):
        if values:
            profile[step] = float(np.nanmean(values))
    return profile


def activation_norm_profile(
    trace_rows: list[dict[str, Any]],
    *,
    condition: str,
) -> dict[int, float]:
    """Aggregate per-step activation_norm for one condition."""

    return _profile_from_metric(
        trace_rows,
        condition=condition,
        metric_name="activation_norm",
    )


def delta_norm_profile(
    trace_rows: list[dict[str, Any]],
    *,
    condition: str,
) -> dict[int, float]:
    """Aggregate per-step activation_delta_norm for one condition."""

    return _profile_from_metric(
        trace_rows,
        condition=condition,
        metric_name="activation_delta_norm",
    )


def _window_mean(profile: dict[int, float], window: tuple[int, int]) -> float:
    vals = [v for s, v in profile.items() if window[0] <= s <= window[1]]
    return float(np.nanmean(vals)) if vals else 0.0


def echo_lock_in_index(
    delta_profile: dict[int, float],
    *,
    pre_window: tuple[int, int],
    post_window: tuple[int, int],
    eps: float = 1e-8,
) -> float:
    """Ratio of post-window to pre-window delta norm (lower indicates lock-in)."""

    pre = _window_mean(delta_profile, pre_window)
    post = _window_mean(delta_profile, post_window)
    if not np.isfinite(pre) or not np.isfinite(post):
        return 0.0
    return float(post / (abs(pre) + eps))


def recovery_index(
    delta_profile: dict[int, float],
    *,
    breaker_window: tuple[int, int],
    post_breaker_window: tuple[int, int],
    eps: float = 1e-8,
) -> float:
    """Normalized rebound in delta norm after breaker window."""

    breaker = _window_mean(delta_profile, breaker_window)
    post = _window_mean(delta_profile, post_breaker_window)
    if not np.isfinite(breaker) or not np.isfinite(post):
        return 0.0
    rebound = max(post - breaker, 0.0)
    return float(rebound / (abs(breaker) + eps))


def _edge_density(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    gy, gx = np.gradient(arr)
    mag = np.sqrt(gx**2 + gy**2)
    return float(np.mean(mag)) if mag.size else 0.0


def _color_hist_entropy(image: np.ndarray, bins: int = 16) -> float:
    arr = np.asarray(image)
    if arr.ndim != 3:
        return 0.0
    entropies: list[float] = []
    for ch in range(arr.shape[2]):
        hist, _ = np.histogram(arr[..., ch], bins=bins, range=(0, 255), density=False)
        probs = hist.astype(np.float64)
        probs = probs / max(probs.sum(), 1.0)
        probs = np.clip(probs, 1e-12, 1.0)
        entropies.append(float(-(probs * np.log(probs)).sum()))
    return float(np.mean(entropies)) if entropies else 0.0


def novelty_proxy(
    images: list[np.ndarray],
    *,
    baseline_images: list[np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute simple image novelty proxies for one condition."""

    if not images:
        return {
            "edge_density_variance": 0.0,
            "color_hist_entropy_mean": 0.0,
            "pixel_mse_mean": 0.0,
            "edge_density_mean": 0.0,
        }

    edges = [_edge_density(img) for img in images]
    entropy = [_color_hist_entropy(img) for img in images]

    if baseline_images is None or len(baseline_images) != len(images):
        ref = images[0]
        mse_vals = [float(np.mean((img.astype(np.float32) - ref.astype(np.float32)) ** 2)) for img in images]
    else:
        mse_vals = [
            float(np.mean((img.astype(np.float32) - base.astype(np.float32)) ** 2))
            for img, base in zip(images, baseline_images, strict=True)
        ]

    return {
        "edge_density_variance": float(np.var(edges)) if edges else 0.0,
        "color_hist_entropy_mean": float(np.mean(entropy)) if entropy else 0.0,
        "pixel_mse_mean": float(np.mean(mse_vals)) if mse_vals else 0.0,
        "edge_density_mean": float(np.mean(edges)) if edges else 0.0,
    }


def plot_delta_norm_over_steps(
    delta_profiles: dict[str, dict[int, float]],
    output_path: Path,
) -> Path:
    """Plot delta-norm trajectories for baseline/echo/echo_breaker."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 4.5))
    ordered = ["baseline", "echo", "echo_breaker"]
    for condition in ordered:
        profile = delta_profiles.get(condition, {})
        if not profile:
            continue
        steps = sorted(profile.keys())
        values = [profile[s] for s in steps]
        plt.plot(steps, values, label=condition)

    plt.xlabel("Step")
    plt.ylabel("Activation delta norm")
    plt.title("Residual delta norm over steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
