from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from neural_bending_toolkit.bends.v2 import BendPlan
from neural_bending_toolkit.bends.v2_diffusion_residual import (
    compile_diffusion_residual_hook,
)

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class _Ctx:
    layer_name: str
    step: int
    cache: dict[str, dict[str, torch.Tensor]]


class _CollectingTracer:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def log(
        self,
        *,
        step: int,
        metric_name: str,
        value: float | int,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.records.append(
            {
                "step": step,
                "metric_name": metric_name,
                "value": float(value),
                "metadata": metadata or {},
            }
        )


def test_residual_echo_uses_previous_cache_value() -> None:
    plan = BendPlan.model_validate(
        {
            "bends": [
                {
                    "name": "echo",
                    "site": {
                        "kind": "diffusion.residual",
                        "layer_names": ["down_blocks.0.resnets.0"],
                    },
                    "actuator": {
                        "type": "residual_echo",
                        "params": {"alpha": 0.5},
                    },
                    "schedule": {"mode": "constant", "strength": 1.0},
                }
            ]
        }
    )
    hook = compile_diffusion_residual_hook(plan)
    x = torch.ones((1, 4), dtype=torch.float32)
    ctx = _Ctx(
        layer_name="down_blocks.0.resnets.0",
        step=1,
        cache={"residual_echo": {"down_blocks.0.resnets.0": torch.full((1, 4), 2.0)}},
    )

    out = hook(x, ctx)

    assert torch.allclose(out, torch.full((1, 4), 2.0))


def test_residual_clamp_caps_tensor_norm() -> None:
    plan = BendPlan.model_validate(
        {
            "bends": [
                {
                    "name": "clamp",
                    "site": {
                        "kind": "diffusion.residual",
                        "allow_all_layers": True,
                    },
                    "actuator": {
                        "type": "residual_clamp",
                        "params": {"max_norm": 1.0},
                    },
                    "schedule": {"mode": "constant", "strength": 1.0},
                }
            ]
        }
    )
    hook = compile_diffusion_residual_hook(plan)

    x = torch.full((1, 4), 3.0)
    out = hook(x, _Ctx(layer_name="any", step=0, cache={"residual_echo": {}}))

    assert out.norm().item() <= 1.0001


def test_residual_leak_is_deterministic_with_manual_seed() -> None:
    plan = BendPlan.model_validate(
        {
            "bends": [
                {
                    "name": "leak",
                    "site": {
                        "kind": "diffusion.residual",
                        "allow_all_layers": True,
                    },
                    "actuator": {
                        "type": "residual_leak",
                        "params": {"leak": 0.25, "noise_scale": 0.0},
                    },
                    "schedule": {"mode": "constant", "strength": 1.0},
                }
            ]
        }
    )
    hook = compile_diffusion_residual_hook(plan)

    x = torch.full((1, 4), 4.0)
    torch.manual_seed(1234)
    out = hook(x, _Ctx(layer_name="any", step=2, cache={"residual_echo": {}}))

    assert torch.allclose(out, torch.full((1, 4), 3.0))


def test_residual_trace_logs_norm_and_delta_norm() -> None:
    tracer = _CollectingTracer()
    plan = BendPlan.model_validate(
        {
            "bends": [
                {
                    "name": "trace-echo",
                    "site": {
                        "kind": "diffusion.residual",
                        "allow_all_layers": True,
                    },
                    "actuator": {
                        "type": "residual_echo",
                        "params": {"alpha": 0.5},
                    },
                    "schedule": {"mode": "constant", "strength": 1.0},
                    "trace": {
                        "metrics": ["activation_norm", "activation_delta_norm"],
                        "sample_every": 1,
                    },
                }
            ]
        }
    )
    hook = compile_diffusion_residual_hook(plan, tracer=tracer)

    x = torch.ones((1, 4), dtype=torch.float32)
    ctx = _Ctx(
        layer_name="layer",
        step=7,
        cache={"residual_echo": {"layer": torch.ones((1, 4), dtype=torch.float32)}},
    )

    _ = hook(x, ctx)

    assert len(tracer.records) == 2
    by_metric = {record["metric_name"]: record for record in tracer.records}
    assert set(by_metric) == {"activation_norm", "activation_delta_norm"}
    assert math.isfinite(by_metric["activation_norm"]["value"])
    assert math.isfinite(by_metric["activation_delta_norm"]["value"])
    assert by_metric["activation_delta_norm"]["value"] > 0
