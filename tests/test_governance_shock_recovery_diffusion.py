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

from neural_bending_toolkit.experiments.governance_shock_recovery_diffusion import (
    GovernanceShockRecoveryDiffusion,
    GovernanceShockRecoveryDiffusionConfig,
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
        hook = kwargs.get("cross_attention_hook")
        value = 0 if hook is None else 32
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

    def pre_intervention_snapshot(self, **_kwargs):
        return None

    def post_intervention_snapshot(self, **_kwargs):
        return None

    def log_event(self, *_args, **_kwargs):
        return None


def test_config_requires_shock_bends() -> None:
    try:
        GovernanceShockRecoveryDiffusionConfig.model_validate({})
    except ValueError:
        return
    raise AssertionError("expected validation error")


def test_experiment_writes_condition_and_comparison_artifacts(monkeypatch, tmp_path) -> None:
    config = GovernanceShockRecoveryDiffusionConfig.model_validate(
        {
            "prompt": ["one prompt"],
            "samples_per_condition": 2,
            "shock": [
                {
                    "name": "shock-gate",
                    "site": {"kind": "diffusion.cross_attention", "allow_all_layers": True},
                    "actuator": {"type": "attention_head_gate", "params": {}},
                    "schedule": {"mode": "constant", "strength": 1.0},
                }
            ],
        }
    )
    experiment = GovernanceShockRecoveryDiffusion(config)
    adapter = _FakeAdapter()
    monkeypatch.setattr(experiment, "_load_adapter", lambda: adapter)

    monkeypatch.setattr(
        "neural_bending_toolkit.experiments.governance_shock_recovery_diffusion.compile_diffusion_cross_attention_hook",
        lambda _plan, tracer=None: (lambda _payload: None) if tracer is not None else None,
    )

    experiment.run(_Context(tmp_path))

    assert (tmp_path / "conditions" / "baseline").exists()
    assert (tmp_path / "conditions" / "shock").exists()
    assert (tmp_path / "conditions" / "shock_counter").exists()

    metrics_path = tmp_path / "comparisons" / "metrics_comparison.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["baseline_vs_shock"]["count"] == 2
    assert payload["baseline_vs_shock"]["mean_mse"] > 0
    assert "metastability" in payload
    assert "basin_shift_proxy" in payload["metastability"]
    assert "tagging" in payload

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["conditions"]["baseline"]["count"] == 2
    assert "metastability" in summary
    assert "tags" in summary
    assert (tmp_path / "comparisons" / "comparison_report.json").exists()
    assert "entropy_over_steps" in summary["artifacts"]
    assert len(adapter.calls) == 6
