from __future__ import annotations

import math

import pytest

from neural_bending_toolkit.bends.v2 import BendPlan
from neural_bending_toolkit.bends.v2_diffusion import (
    compile_diffusion_cross_attention_hook,
)

torch = pytest.importorskip("torch")


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


def _payload() -> dict[str, object]:
    query = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[0.3, 0.7], [0.2, 0.8]],
        ],
        dtype=torch.float32,
    )
    key = query.clone()
    value = torch.ones_like(query)
    attention_probs = torch.softmax(torch.tensor(
        [
            [[2.0, 1.0], [1.0, 2.0]],
            [[1.5, 1.0], [0.5, 2.0]],
        ],
        dtype=torch.float32,
    ), dim=-1)
    return {
        "layer": "down_blocks.0.attentions.0.transformer_blocks.0.attn2",
        "step": 5,
        "query": query,
        "key": key,
        "value": value,
        "attention_probs": attention_probs,
    }


def test_compile_hook_head_gate_changes_probs_shape_consistently() -> None:
    plan = BendPlan.model_validate(
        {
            "bends": [
                {
                    "name": "gate-head-0",
                    "site": {
                        "kind": "diffusion.cross_attention",
                        "allow_all_layers": True,
                        "head_indices": [0],
                    },
                    "actuator": {
                        "type": "attention_head_gate",
                        "params": {"scale": 0.0, "num_heads": 2},
                    },
                    "schedule": {"mode": "constant", "strength": 1.0},
                }
            ]
        }
    )
    hook = compile_diffusion_cross_attention_hook(plan)
    payload = _payload()

    modified = hook(payload)

    assert modified is not None
    before = payload["attention_probs"]
    after = modified["attention_probs"]
    assert before.shape == after.shape
    assert torch.allclose(after[0], torch.zeros_like(after[0]))
    assert torch.allclose(after[1], before[1])


def test_compile_hook_respects_timestep_window() -> None:
    plan = BendPlan.model_validate(
        {
            "bends": [
                {
                    "name": "windowed-gate",
                    "site": {
                        "kind": "diffusion.cross_attention",
                        "allow_all_layers": True,
                        "timestep_start": 10,
                        "timestep_end": 20,
                    },
                    "actuator": {
                        "type": "attention_head_gate",
                        "params": {"scale": 0.0},
                    },
                    "schedule": {"mode": "window", "strength": 1.0},
                }
            ]
        }
    )
    hook = compile_diffusion_cross_attention_hook(plan)

    payload = _payload()
    payload["step"] = 5

    assert hook(payload) is None


def test_compile_hook_logs_finite_attention_entropy() -> None:
    tracer = _CollectingTracer()
    plan = BendPlan.model_validate(
        {
            "bends": [
                {
                    "name": "trace-entropy",
                    "site": {
                        "kind": "diffusion.cross_attention",
                        "allow_all_layers": True,
                    },
                    "actuator": {"type": "noop"},
                    "schedule": {"mode": "constant", "strength": 1.0},
                    "trace": {
                        "metrics": ["attention_entropy"],
                        "sample_every": 1,
                    },
                }
            ]
        }
    )

    hook = compile_diffusion_cross_attention_hook(plan, tracer=tracer)
    _ = hook(_payload())

    assert len(tracer.records) == 1
    value = tracer.records[0]["value"]
    metadata = tracer.records[0]["metadata"]
    assert isinstance(value, float)
    assert math.isfinite(value)
    assert isinstance(metadata, dict)
    assert "localizability" in metadata
