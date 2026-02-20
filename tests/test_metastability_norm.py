from __future__ import annotations

import math

from neural_bending_toolkit.analysis.bend_tagging_norm import (
    score_and_tag_norm_metastability,
)
from neural_bending_toolkit.analysis.metastability_norm import (
    collapse_index,
    recovery_index,
    stability_proxy,
    variance_profile,
)


def test_variance_profile_and_indices() -> None:
    rows = [
        {"step": 1, "metric_name": "norm_output_var", "value": 1.0, "metadata": {"condition": "norm_recovery"}},
        {"step": 2, "metric_name": "norm_output_var", "value": 0.4, "metadata": {"condition": "norm_recovery"}},
        {"step": 3, "metric_name": "norm_output_var", "value": 0.8, "metadata": {"condition": "norm_recovery"}},
    ]
    profile = variance_profile(rows, condition="norm_recovery")
    assert profile == {1: 1.0, 2: 0.4, 3: 0.8}

    collapse = collapse_index(profile, pre_window=(1, 1), during_window=(2, 2))
    recovery = recovery_index(profile, during_window=(2, 2), post_window=(3, 3))
    stability = stability_proxy(profile)

    assert collapse > 0
    assert recovery > 0
    assert math.isclose(stability, 1.0)


def test_norm_tagging_rules_disruptive_when_unstable() -> None:
    result = score_and_tag_norm_metastability(
        collapse_index_value=0.6,
        recovery_index_value=0.1,
        stability_proxy_value=0.5,
        final_basin_shift=1200.0,
    )

    assert "disruptive" in result.tags
