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

from neural_bending_toolkit.experiments.norm_drift_collapse_recovery_diffusion import (
    NormDriftCollapseRecoveryDiffusion,
    NormDriftCollapseRecoveryDiffusionConfig,
)


class _FakeImage:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr.astype(np.uint8)

    def save(self, path):
        path.write_bytes(b"fake-png")

    def __array__(self, dtype=None):
        return self._arr.astype(dtype) if dtype else self._arr


class _FakeOutput:
    def __init__(self, value: int) -> None:
        self.images = [_FakeImage(np.full((4, 4, 3), value, dtype=np.uint8))]
        self.attention_heatmaps = {}


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, _prompt: str, **kwargs):
        self.calls.append(kwargs)
        hook = kwargs.get("norm_hook")
        if hook is None:
            value = 100
        else:
            marker = float(hook({"layer": "norm", "step": 10, "tensor": np.array([1.0])})[0])
            value = 60 if marker < 0 else 80
        return _FakeOutput(value)

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


def test_experiment_writes_norm_comparison_and_summary(monkeypatch, tmp_path) -> None:
    config = NormDriftCollapseRecoveryDiffusionConfig.model_validate(
        {
            "prompt": ["one prompt"],
            "samples_per_condition": 2,
            "collapse": [
                {
                    "name": "collapse",
                    "site": {"kind": "diffusion.norm", "allow_all_layers": True, "timestep_start": 8, "timestep_end": 12},
                    "actuator": {"type": "norm_gain_drift", "params": {"scale": 1.0}},
                    "schedule": {"mode": "window", "strength": 1.0},
                    "trace": {"metrics": ["norm_output_var"]},
                }
            ],
            "counter": [
                {
                    "name": "counter",
                    "site": {"kind": "diffusion.norm", "allow_all_layers": True, "timestep_start": 13, "timestep_end": 20},
                    "actuator": {"type": "norm_bias_shift", "params": {"bias": -2.0}},
                    "schedule": {"mode": "window", "strength": 1.0},
                    "trace": {"metrics": ["norm_output_var"]},
                }
            ],
        }
    )
    experiment = NormDriftCollapseRecoveryDiffusion(config)
    adapter = _FakeAdapter()
    monkeypatch.setattr(experiment, "_load_adapter", lambda: adapter)

    def _fake_compile(plan, tracer=None):
        has_counter = any(b.name == "counter" for b in plan.bends)

        def _hook(_payload):
            if tracer:
                tracer.log(step=10, metric_name="norm_output_var", value=0.02 if not has_counter else 0.05)
                tracer.log(step=20, metric_name="norm_output_var", value=0.03 if not has_counter else 0.08)
            return np.array([-1.0 if has_counter else 1.0])

        return _hook

    monkeypatch.setattr(
        "neural_bending_toolkit.experiments.norm_drift_collapse_recovery_diffusion.compile_diffusion_norm_hook",
        _fake_compile,
    )

    experiment.run(_Context(tmp_path))

    payload = json.loads((tmp_path / "comparisons" / "norm_metrics_comparison.json").read_text())
    assert "collapse_index" in payload
    assert "recovery_index" in payload
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["conditions"]["baseline"]["count"] == 2
    assert len(adapter.calls) == 6
