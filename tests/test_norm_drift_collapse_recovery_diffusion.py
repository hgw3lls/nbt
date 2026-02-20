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
            marker = float(hook("ignored", object()))
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
        }
    )
    experiment = NormDriftCollapseRecoveryDiffusion(config)
    adapter = _FakeAdapter()
    monkeypatch.setattr(experiment, "_load_adapter", lambda: adapter)

    def _fake_compile(plan, tracer=None):
        has_recovery = any(bend.name == "norm-recovery-late-gain-counter" for bend in plan.bends)

        def _hook(_x, _ctx):
            if tracer is not None:
                tracer.log(
                    step=10,
                    metric_name="norm_output_var",
                    value=0.02 if not has_recovery else 0.05,
                    metadata={"layer": "down.norm"},
                )
                tracer.log(
                    step=20,
                    metric_name="norm_output_var",
                    value=0.03 if not has_recovery else 0.08,
                    metadata={"layer": "down.norm"},
                )
            return 1.0 if not has_recovery else -1.0

        return _hook

    monkeypatch.setattr(
        "neural_bending_toolkit.experiments.norm_drift_collapse_recovery_diffusion.compile_diffusion_norm_hook",
        _fake_compile,
    )

    def _fake_plot(_profiles, output_path):
        output_path.write_bytes(b"png")
        return output_path

    monkeypatch.setattr(
        "neural_bending_toolkit.experiments.norm_drift_collapse_recovery_diffusion.plot_norm_variance_over_steps",
        _fake_plot,
    )

    experiment.run(_Context(tmp_path))

    assert (tmp_path / "conditions" / "baseline").exists()
    assert (tmp_path / "conditions" / "norm_collapse").exists()
    assert (tmp_path / "conditions" / "norm_recovery").exists()

    payload = json.loads((tmp_path / "comparisons" / "norm_metrics_comparison.json").read_text())
    assert "norm_output_var_profiles" in payload
    assert "collapse_index" in payload
    assert "recovery_index" in payload
    assert "stability_proxy" in payload
    assert "tagging" in payload
    assert payload["image_difference_proxy"]["norm_collapse"]["mean_mse"] > 0

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["conditions"]["baseline"]["count"] == 2
    assert "tags" in summary
    assert "tagging" in summary
    assert (tmp_path / "comparisons" / "norm_variance_over_steps.png").exists()
    assert len(adapter.calls) == 6
