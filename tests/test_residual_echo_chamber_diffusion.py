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
        condition = getattr(hook, "condition", "baseline") if hook is not None else "baseline"
        value_by_condition = {
            "baseline": 16,
            "echo": 48,
            "echo_breaker": 32,
        }
        return _FakeOutput(value_by_condition.get(condition, 16))

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

    def post_intervention_snapshot(self, **_kwargs):
        return None


def test_residual_echo_chamber_writes_expected_artifacts(monkeypatch, tmp_path) -> None:
    config = ResidualEchoChamberDiffusionConfig.model_validate(
        {
            "prompt": ["one prompt"],
            "samples_per_condition": 2,
            "num_inference_steps": 30,
        }
    )
    experiment = ResidualEchoChamberDiffusion(config)
    adapter = _FakeAdapter()
    monkeypatch.setattr(experiment, "_load_adapter", lambda: adapter)

    def _fake_compile(_plan, tracer=None):
        def _hook(x, _ctx):
            return x

        _hook.condition = getattr(tracer, "condition", "echo")
        return _hook

    monkeypatch.setattr(
        "neural_bending_toolkit.experiments.residual_echo_chamber_diffusion.compile_diffusion_residual_hook",
        _fake_compile,
    )

    experiment.run(_Context(tmp_path))

    assert (tmp_path / "conditions" / "baseline").exists()
    assert (tmp_path / "conditions" / "echo").exists()
    assert (tmp_path / "conditions" / "echo_breaker").exists()

    metrics_path = tmp_path / "comparisons" / "residual_metrics_comparison.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["baseline_vs_echo"]["count"] == 2
    assert payload["echo_vs_echo_breaker"]["mean_mse"] > 0
    assert "residual_metrics" in payload

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["conditions"]["baseline"]["count"] == 2
    assert "residual_echo_chamber" in summary["tags"]
    assert "echo_lock_in_index" in summary["residual_metrics"]
    assert "recovery_index" in summary["residual_metrics"]
    assert "delta_norm_over_steps" in summary["artifacts"]
    assert len(adapter.calls) == 6
