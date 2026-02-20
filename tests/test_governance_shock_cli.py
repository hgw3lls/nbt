import sys
import types
from pathlib import Path

from typer.testing import CliRunner

if "matplotlib" not in sys.modules:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

from neural_bending_toolkit.cli import app


def test_cli_list_includes_governance_shock_experiment() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "governance-shock-recovery-diffusion" in result.output


def test_cli_run_governance_shock_routes_to_runner(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shock:\n  - name: x\n    site:\n      kind: diffusion.cross_attention\n      allow_all_layers: true\n    actuator:\n      type: noop\n      params: {}\n    schedule:\n      mode: constant\n      strength: 1\n", encoding="utf-8")

    def _fake_run_experiment(experiment: str, config: Path, registry):
        assert experiment == "governance-shock-recovery-diffusion"
        assert config == config_path
        assert "governance-shock-recovery-diffusion" in registry.list_experiments()
        return tmp_path / "runs" / "fake"

    monkeypatch.setattr("neural_bending_toolkit.cli.run_experiment", _fake_run_experiment)

    result = runner.invoke(
        app,
        [
            "run",
            "governance-shock-recovery-diffusion",
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert "Run completed:" in result.output
