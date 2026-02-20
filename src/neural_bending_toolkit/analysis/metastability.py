"""Metastability metrics for shock/recovery diffusion analyses."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _mean_by_step(
    traces: list[dict[str, Any]],
    *,
    metric_name: str,
) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in traces:
        if str(row.get("metric_name", "")).strip().lower() != metric_name.lower():
            continue
        step = row.get("step")
        value = row.get("value")
        if not isinstance(step, int) or not isinstance(value, (int, float)):
            continue
        grouped[step].append(float(value))

    return [float(np.mean(grouped[step])) for step in sorted(grouped)]


def compute_attention_entropy_profile(traces: list[dict[str, Any]]) -> list[float]:
    """Return mean attention entropy per step from trace rows."""

    return _mean_by_step(traces, metric_name="attention_entropy")


def compute_attention_topk_mass_profile(traces: list[dict[str, Any]]) -> list[float]:
    """Return mean attention top-k mass per step from trace rows."""

    return _mean_by_step(traces, metric_name="attention_topk_mass")


def recovery_index(profile: list[float], shock_window: tuple[int, int]) -> float:
    """Estimate rebound after shock relative to pre-shock displacement."""

    if len(profile) < 3:
        return 0.0

    start, end = shock_window
    start = max(0, int(start))
    end = min(len(profile) - 1, int(end))
    if end < start:
        return 0.0

    pre = profile[:start] or [profile[start]]
    during = profile[start : end + 1]
    post = profile[end + 1 :] or [profile[end]]

    pre_mean = float(np.mean(pre))
    during_mean = float(np.mean(during))
    post_mean = float(np.mean(post))
    eps = 1e-8
    return float((post_mean - during_mean) / (abs(during_mean - pre_mean) + eps))


def concentration_collapse_index(
    topk_mass_profile: list[float],
    shock_window: tuple[int, int],
) -> float:
    """Measure concentration increase during shock against pre-shock baseline."""

    if len(topk_mass_profile) < 2:
        return 0.0
    start, end = shock_window
    start = max(0, int(start))
    end = min(len(topk_mass_profile) - 1, int(end))
    if end < start:
        return 0.0

    pre = topk_mass_profile[:start] or [topk_mass_profile[start]]
    during = topk_mass_profile[start : end + 1]
    eps = 1e-8
    return float((float(np.mean(during)) - float(np.mean(pre))) / (abs(float(np.mean(pre))) + eps))


def _pairwise_mse(arrays: dict[str, list[np.ndarray]]) -> dict[str, float]:
    names = sorted(arrays)
    out: dict[str, float] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            lhs = arrays[left]
            rhs = arrays[right]
            if len(lhs) != len(rhs) or not lhs:
                continue
            mse_vals = [
                float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
                for a, b in zip(lhs, rhs, strict=True)
            ]
            out[f"{left}_vs_{right}"] = float(np.mean(mse_vals))
    return out


def _edge_density(image: np.ndarray) -> float:
    gray = image.astype(np.float32).mean(axis=-1) if image.ndim == 3 else image.astype(np.float32)
    gx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    return float((gx + gy) / 2.0)


def _histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    hist_a, _ = np.histogram(a_flat, bins=32, range=(0, 255), density=True)
    hist_b, _ = np.histogram(b_flat, bins=32, range=(0, 255), density=True)
    return float(np.mean(np.abs(hist_a - hist_b)))


def basin_shift_proxy(images_or_latents: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    """Compute robust basin-shift proxies with graceful fallbacks.

    Returns pairwise MSE always, plus optional histogram/edge/ssim proxies.
    """

    pairwise_mse = _pairwise_mse(images_or_latents)
    names = sorted(images_or_latents)

    pairwise_hist: dict[str, float] = {}
    pairwise_edge: dict[str, float] = {}
    pairwise_ssim: dict[str, float] = {}
    limitations: list[str] = []

    try:
        from skimage.metrics import structural_similarity

        has_ssim = True
    except Exception:
        structural_similarity = None
        has_ssim = False
        limitations.append("SSIM unavailable; skimage not installed")

    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            lhs = images_or_latents[left]
            rhs = images_or_latents[right]
            if len(lhs) != len(rhs) or not lhs:
                continue
            key = f"{left}_vs_{right}"
            hist_vals: list[float] = []
            edge_vals: list[float] = []
            ssim_vals: list[float] = []
            for a, b in zip(lhs, rhs, strict=True):
                hist_vals.append(_histogram_distance(a, b))
                edge_vals.append(abs(_edge_density(a) - _edge_density(b)))
                if has_ssim and structural_similarity is not None:
                    gray_a = a.astype(np.float32).mean(axis=-1) if a.ndim == 3 else a.astype(np.float32)
                    gray_b = b.astype(np.float32).mean(axis=-1) if b.ndim == 3 else b.astype(np.float32)
                    data_range = max(float(gray_a.max() - gray_a.min()), 1.0)
                    ssim_vals.append(
                        float(structural_similarity(gray_a, gray_b, data_range=data_range))
                    )
            pairwise_hist[key] = float(np.mean(hist_vals)) if hist_vals else 0.0
            pairwise_edge[key] = float(np.mean(edge_vals)) if edge_vals else 0.0
            if ssim_vals:
                pairwise_ssim[key] = float(np.mean(ssim_vals))

    return {
        "pairwise_mse": pairwise_mse,
        "pairwise_histogram_l1": pairwise_hist,
        "pairwise_edge_density_delta": pairwise_edge,
        "pairwise_ssim": pairwise_ssim,
        "limitations": limitations,
    }
