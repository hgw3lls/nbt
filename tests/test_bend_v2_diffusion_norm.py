from __future__ import annotations

import math

import pytest

from neural_bending_toolkit.bends.v2 import BendPlan
from neural_bending_toolkit.bends.v2_diffusion_norm import compile_diffusion_norm_hook

torch = pytest.importorskip("torch")


class _CollectingTracer:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def log(self, *, step: int, metric_name: str, value: float | int, metadata: dict[str, object] | None = None) -> None:
        self.records.append({"step": step, "metric_name": metric_name, "value": float(value), "metadata": metadata or {}})


def test_norm_gain_drift_matches_layer_and_changes_tensor() -> None:
    plan = BendPlan.model_validate({"bends": [{"name": "gain", "site": {"kind": "diffusion.norm", "layer_regex": "down_blocks.*norm", "timestep_start": 2, "timestep_end": 4}, "actuator": {"type": "norm_gain_drift"}, "schedule": {"mode": "constant", "strength": 0.5}}]})
    hook = compile_diffusion_norm_hook(plan)
    x = torch.ones((1, 4, 2, 2), dtype=torch.float32)
    y = hook({"layer": "down_blocks.0.norm1", "step": 3, "tensor": x})
    assert y.shape == x.shape
    assert torch.allclose(y, x * 1.5)


def test_norm_hook_respects_pulse_schedule() -> None:
    plan = BendPlan.model_validate({"bends": [{"name": "pulse-bias", "site": {"kind": "diffusion.norm", "allow_all_layers": True}, "actuator": {"type": "norm_bias_shift"}, "schedule": {"mode": "pulse", "strength": 2.0, "period": 4, "duty": 0.25}}]})
    hook = compile_diffusion_norm_hook(plan)
    x = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    active = hook({"layer": "norm", "step": 0, "tensor": x})
    inactive = hook({"layer": "norm", "step": 2, "tensor": x})
    assert torch.allclose(active, torch.full_like(x, 2.0))
    assert inactive is None


def test_norm_hook_traces_mean_var() -> None:
    tracer = _CollectingTracer()
    plan = BendPlan.model_validate({"bends": [{"name": "noise-trace", "site": {"kind": "diffusion.norm", "allow_all_layers": True}, "actuator": {"type": "activation_noise", "params": {"sigma": 0.0}}, "schedule": {"mode": "constant", "strength": 1.0}, "trace": {"metrics": ["norm_output_mean", "norm_output_var"], "sample_every": 1}}]})
    hook = compile_diffusion_norm_hook(plan, tracer=tracer)
    x = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    _ = hook({"layer": "any.norm", "step": 1, "tensor": x})
    assert len(tracer.records) == 2
    metric_names = {record["metric_name"] for record in tracer.records}
    assert metric_names == {"norm_output_mean", "norm_output_var"}
    assert all(math.isfinite(record["value"]) for record in tracer.records)
