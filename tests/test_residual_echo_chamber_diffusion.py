from __future__ import annotations

import sys
import types

if "matplotlib" not in sys.modules:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

import json

import numpy as np

from neural_bending_toolkit.experiments.residual_echo_chamber_diffusion import (
    ResidualEchoChamberDiffusion,
    ResidualEchoChamberDiffusionConfig,
)


class _FakeImage:
    def __init__(self, value: int) -> None:
        self._arr = np.full((4, 4, 3), value, dtype=np.uint8)

    def save(self, path):
        path.write_bytes(b"fake-png")

    def __array__(self, dtype=None):
        return self._arr.astype(dtype) if dtype else self._arr


class _FakeOutput:
    def __init__(self, value: int) -> None:
        self.images = [_FakeImage(value)]
        self.attention_heatmaps = {}


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, _prompt: str, **kwargs):
        self.calls.append(kwargs)
        hook = kwargs.get("residual_hook")
        condition = getattr(hook, "condition", "baseline") if hook else "baseline"
        return _FakeOutput({"baseline": 16, "echo": 48, "echo_breaker": 32}.get(condition, 16))

    def save_artifacts(self, output, artifacts_dir, *, prefix: str):
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        image_path = artifacts_dir / f"{prefix}_image_0.png"
        output.images[0].save(image_path)
        return [image_path]


class _Context:
    def __init__(self, run_dir) -> None:
        self.run_dir = run_dir

    def log_metric(self, **_kwargs):
        return None


def test_residual_echo_chamber_writes_expected_artifacts(monkeypatch, tmp_path) -> None:
    config = ResidualEchoChamberDiffusionConfig.model_validate(
        {
            "prompt": ["one prompt"],
            "samples_per_condition": 2,
            "echo": [{"name": "echo", "site": {"kind": "diffusion.residual", "allow_all_layers": True, "timestep_start": 10, "timestep_end": 20}, "actuator": {"type": "residual_echo", "params": {"alpha": 0.5}}, "schedule": {"mode": "window", "strength": 1.0}, "trace": {"metrics": ["activation_delta_norm"]}}],
            "counter": [{"name": "counter", "site": {"kind": "diffusion.residual", "allow_all_layers": True, "timestep_start": 21, "timestep_end": 28}, "actuator": {"type": "residual_clamp", "params": {"max_norm": 1.0}}, "schedule": {"mode": "window", "strength": 1.0}, "trace": {"metrics": ["activation_delta_norm"]}}],
        }
    )
    experiment = ResidualEchoChamberDiffusion(config)
    adapter = _FakeAdapter()
    monkeypatch.setattr(experiment, "_load_adapter", lambda: adapter)

    def _fake_compile(_plan, tracer=None):
        def _hook(_payload):
            return None

        _hook.condition = tracer.metadata["condition"] if tracer else "echo"
        return _hook

    monkeypatch.setattr(
        "neural_bending_toolkit.experiments.residual_echo_chamber_diffusion.compile_diffusion_residual_hook",
        _fake_compile,
    )

    experiment.run(_Context(tmp_path))

    payload = json.loads((tmp_path / "comparisons" / "residual_metrics_comparison.json").read_text(encoding="utf-8"))
    assert payload["baseline_vs_echo"]["count"] == 2
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "echo_lock_in_index" in summary["residual_metrics"]
    assert len(adapter.calls) == 6
