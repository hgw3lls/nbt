from pathlib import Path

import pytest

from neural_bending_toolkit.bends import BendPlan
from neural_bending_toolkit.bends.v2 import bend_localizability_label


def test_bend_plan_round_trip_from_yaml_template() -> None:
    yaml = pytest.importorskip("yaml")
    path = Path("templates/bend_v2.example.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    plan = BendPlan.model_validate(raw)
    assert len(plan.bends) == 1
    assert plan.bends[0].site.kind == "diffusion.cross_attention"
    assert plan.bends[0].actuator.type == "attention_probs_temperature"

    dumped = plan.model_dump()
    reparsed = BendPlan.model_validate(dumped)
    assert reparsed == plan


def test_pulse_schedule_requires_period_and_duty() -> None:
    with pytest.raises(ValueError):
        BendPlan.model_validate(
            {
                "bends": [
                    {
                        "name": "pulse-missing-fields",
                        "site": {
                            "kind": "diffusion.cross_attention",
                            "allow_all_layers": True,
                        },
                        "actuator": {"type": "noop"},
                        "schedule": {"mode": "pulse", "strength": 0.5},
                    }
                ]
            }
        )


def test_site_requires_selector_or_allow_all_layers() -> None:
    with pytest.raises(ValueError, match="requires layer_regex or layer_names"):
        BendPlan.model_validate(
            {
                "bends": [
                    {
                        "name": "missing-selector",
                        "site": {"kind": "diffusion.cross_attention"},
                        "actuator": {"type": "noop"},
                        "schedule": {"mode": "constant", "strength": 0.5},
                    }
                ]
            }
        )


def test_site_rejects_negative_head_indices() -> None:
    with pytest.raises(ValueError, match="head_indices"):
        BendPlan.model_validate(
            {
                "bends": [
                    {
                        "name": "negative-head",
                        "site": {
                            "kind": "diffusion.cross_attention",
                            "allow_all_layers": True,
                            "head_indices": [-1],
                        },
                        "actuator": {"type": "noop"},
                        "schedule": {"mode": "constant", "strength": 0.5},
                    }
                ]
            }
        )


def test_site_rejects_invalid_timestep_order() -> None:
    with pytest.raises(ValueError, match="timestep_start"):
        BendPlan.model_validate(
            {
                "bends": [
                    {
                        "name": "bad-window",
                        "site": {
                            "kind": "diffusion.cross_attention",
                            "allow_all_layers": True,
                            "timestep_start": 10,
                            "timestep_end": 2,
                        },
                        "actuator": {"type": "noop"},
                        "schedule": {"mode": "constant", "strength": 0.5},
                    }
                ]
            }
        )


def test_schedule_rejects_non_finite_strength() -> None:
    with pytest.raises(ValueError, match="strength must be finite"):
        BendPlan.model_validate(
            {
                "bends": [
                    {
                        "name": "nan-strength",
                        "site": {
                            "kind": "diffusion.cross_attention",
                            "allow_all_layers": True,
                        },
                        "actuator": {"type": "noop"},
                        "schedule": {"mode": "constant", "strength": float("nan")},
                    }
                ]
            }
        )


def test_localizability_label_format() -> None:
    plan = BendPlan.model_validate(
        {
            "bends": [
                {
                    "name": "loc",
                    "site": {
                        "kind": "diffusion.cross_attention",
                        "layer_regex": "down_blocks.*attn2",
                        "head_indices": [0, 3],
                        "timestep_start": 2,
                        "timestep_end": 6,
                    },
                    "actuator": {"type": "noop"},
                    "schedule": {"mode": "constant", "strength": 0.4},
                }
            ]
        }
    )

    label = bend_localizability_label(plan.bends[0])
    assert (
        label
        == "diffusion.cross_attention: layer=down_blocks.*attn2 head=[0,3] steps=2..6"
    )


def test_norm_site_and_actuator_validate() -> None:
    plan = BendPlan.model_validate(
        {
            "bends": [
                {
                    "name": "norm",
                    "site": {
                        "kind": "diffusion.norm",
                        "allow_all_layers": True,
                    },
                    "actuator": {"type": "norm_gain_drift", "params": {}},
                    "schedule": {"mode": "constant", "strength": 0.5},
                    "trace": {"metrics": ["norm_output_mean", "activation_snr"]},
                }
            ]
        }
    )

    assert plan.bends[0].site.kind == "diffusion.norm"
    assert plan.bends[0].actuator.type == "norm_gain_drift"
