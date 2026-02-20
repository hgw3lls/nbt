from __future__ import annotations

import sys
import types

if "matplotlib" not in sys.modules:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

import numpy as np

from neural_bending_toolkit.analysis.metastability_residual import (
    activation_norm_profile,
    delta_norm_profile,
    echo_lock_in_index,
    novelty_proxy,
    recovery_index,
)


def test_profiles_and_indices_compute_expected_shapes() -> None:
    traces = [
        {"step": 0, "metric_name": "activation_norm", "value": 2.0, "metadata": {"condition": "echo"}},
        {"step": 1, "metric_name": "activation_norm", "value": 3.0, "metadata": {"condition": "echo"}},
        {"step": 0, "metric_name": "activation_delta_norm", "value": 4.0, "metadata": {"condition": "echo"}},
        {"step": 1, "metric_name": "activation_delta_norm", "value": 1.0, "metadata": {"condition": "echo"}},
        {"step": 2, "metric_name": "activation_delta_norm", "value": 2.0, "metadata": {"condition": "echo_breaker"}},
        {"step": 3, "metric_name": "activation_delta_norm", "value": 4.0, "metadata": {"condition": "echo_breaker"}},
    ]

    act = activation_norm_profile(traces, condition="echo")
    delta = delta_norm_profile(traces, condition="echo")

    assert act == {0: 2.0, 1: 3.0}
    assert delta == {0: 4.0, 1: 1.0}

    lock = echo_lock_in_index(delta, pre_window=(0, 0), post_window=(1, 1))
    assert lock < 1.0

    rec = recovery_index(
        delta_norm_profile(traces, condition="echo_breaker"),
        breaker_window=(2, 2),
        post_breaker_window=(3, 3),
    )
    assert rec > 0


def test_novelty_proxy_exposes_fallback_metrics() -> None:
    img_a = np.zeros((8, 8, 3), dtype=np.uint8)
    img_b = np.full((8, 8, 3), 255, dtype=np.uint8)
    metrics = novelty_proxy([img_a, img_b], baseline_images=[img_a, img_a])

    assert "edge_density_variance" in metrics
    assert "color_hist_entropy_mean" in metrics
    assert "pixel_mse_mean" in metrics
    assert metrics["pixel_mse_mean"] > 0
