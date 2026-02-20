import sys
import types

if "matplotlib" not in sys.modules:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

import numpy as np

from neural_bending_toolkit.analysis.metastability import (
    basin_shift_proxy,
    compute_attention_entropy_profile,
    concentration_collapse_index,
    recovery_index,
)


def test_attention_entropy_profile_groups_by_step() -> None:
    traces = [
        {"step": 0, "metric_name": "attention_entropy", "value": 1.0},
        {"step": 0, "metric_name": "attention_entropy", "value": 3.0},
        {"step": 1, "metric_name": "attention_entropy", "value": 2.0},
    ]
    assert compute_attention_entropy_profile(traces) == [2.0, 2.0]


def test_recovery_and_collapse_indices() -> None:
    ent = [1.0, 1.0, 0.5, 0.6, 0.9, 1.0]
    topk = [0.4, 0.4, 0.7, 0.75, 0.5, 0.45]

    assert recovery_index(ent, (2, 3)) > 0
    assert concentration_collapse_index(topk, (2, 3)) > 0


def test_basin_shift_proxy_returns_mse_and_fallback_fields() -> None:
    baseline = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    shock = [np.full((4, 4, 3), 16, dtype=np.uint8) for _ in range(2)]
    counter = [np.full((4, 4, 3), 8, dtype=np.uint8) for _ in range(2)]

    payload = basin_shift_proxy(
        {
            "baseline": baseline,
            "shock": shock,
            "shock_counter": counter,
        }
    )

    assert payload["pairwise_mse"]["baseline_vs_shock"] > 0
    assert "pairwise_histogram_l1" in payload
    assert "pairwise_edge_density_delta" in payload
