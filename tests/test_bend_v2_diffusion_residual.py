from __future__ import annotations

import math

import pytest

from neural_bending_toolkit.bends.v2 import BendPlan
from neural_bending_toolkit.bends.v2_diffusion_residual import compile_diffusion_residual_hook

torch = pytest.importorskip("torch")


class _CollectingTracer:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def log(self, *, step: int, metric_name: str, value: float | int, metadata: dict[str, object] | None = None) -> None:
        self.records.append({"step": step, "metric_name": metric_name, "value": float(value), "metadata": metadata or {}})


def test_residual_echo_uses_previous_cache_value() -> None:
    plan = BendPlan.model_validate({"bends": [{"name": "echo", "site": {"kind": "diffusion.residual", "layer_names": ["down_blocks.0.resnets.0"]}, "actuator": {"type": "residual_echo", "params": {"alpha": 0.5}}, "schedule": {"mode": "constant", "strength": 1.0}}]})
    hook = compile_diffusion_residual_hook(plan)
    x = torch.ones((1, 4), dtype=torch.float32)
    out = hook({"layer": "down_blocks.0.resnets.0", "step": 1, "tensor": x, "prev": torch.full((1, 4), 2.0)})
    assert torch.allclose(out, torch.full((1, 4), 2.0))


def test_residual_clamp_caps_tensor_norm() -> None:
    plan = BendPlan.model_validate({"bends": [{"name": "clamp", "site": {"kind": "diffusion.residual", "allow_all_layers": True}, "actuator": {"type": "residual_clamp", "params": {"max_norm": 1.0}}, "schedule": {"mode": "constant", "strength": 1.0}}]})
    hook = compile_diffusion_residual_hook(plan)
    out = hook({"layer": "any", "step": 0, "tensor": torch.full((1, 4), 3.0)})
    assert out.norm().item() <= 1.0001


def test_residual_trace_logs_norm_and_delta_norm() -> None:
    tracer = _CollectingTracer()
    plan = BendPlan.model_validate({"bends": [{"name": "trace-echo", "site": {"kind": "diffusion.residual", "allow_all_layers": True}, "actuator": {"type": "residual_echo", "params": {"alpha": 0.5}}, "schedule": {"mode": "constant", "strength": 1.0}, "trace": {"metrics": ["activation_norm", "activation_delta_norm"], "sample_every": 1}}]})
    hook = compile_diffusion_residual_hook(plan, tracer=tracer)
    x = torch.ones((1, 4), dtype=torch.float32)
    _ = hook({"layer": "layer", "step": 7, "tensor": x, "prev": torch.ones((1, 4), dtype=torch.float32)})
    assert len(tracer.records) == 2
    by_metric = {record["metric_name"]: record for record in tracer.records}
    assert set(by_metric) == {"activation_norm", "activation_delta_norm"}
    assert math.isfinite(by_metric["activation_norm"]["value"])
    assert math.isfinite(by_metric["activation_delta_norm"]["value"])
