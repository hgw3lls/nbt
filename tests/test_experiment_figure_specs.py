import sys
import types

if "matplotlib" not in sys.modules:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

from pathlib import Path

from pydantic import BaseModel

from neural_bending_toolkit.experiment import Experiment


class _DummyConfig(BaseModel):
    value: int = 1


class _DummyExperiment(Experiment):
    name = "dummy"
    config_model = _DummyConfig

    def recommended_figure_specs(self, run_dir: Path) -> list[dict]:
        return [
            {
                "figure_id": "dummy_fig",
                "title": "Dummy Figure",
                "input_run_dirs": [str(run_dir)],
                "plot_type": "divergence_bar_chart",
                "inputs": {},
                "output_format": {"png": True, "pdf": True},
                "caption_template_variables": {"limit": "x", "threshold": "y"},
            }
        ]

    def run(self, context):
        return None


class _BaseFigureExperiment(Experiment):
    name = "base-figure"
    config_model = _DummyConfig

    def run(self, context):
        return None


def test_emit_figure_specs_writes_yaml(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    exp = _DummyExperiment(_DummyConfig())
    emitted = exp.emit_figure_specs(run_dir)

    assert len(emitted) == 1
    assert emitted[0].exists()
    assert emitted[0].suffix == ".yaml"


def test_base_recommended_specs_add_entropy_figure_when_metric_present(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").write_text(
        '{"step": 0, "metric_name": "attention_entropy", "value": 0.7}\n',
        encoding="utf-8",
    )

    exp = _BaseFigureExperiment(_DummyConfig())
    specs = exp.recommended_figure_specs(run_dir)
    figure_ids = {spec["figure_id"] for spec in specs}

    assert f"{run_dir.name}_attention_entropy" in figure_ids
