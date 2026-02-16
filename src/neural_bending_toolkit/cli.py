"""CLI entrypoint for neural_bending_toolkit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from neural_bending_toolkit.analysis.geopolitical_report import (
    compare_geopolitical_runs,
    generate_geopolitical_report,
)
from neural_bending_toolkit.analysis.report import generate_markdown_report
from neural_bending_toolkit.registry import ExperimentRegistry
from neural_bending_toolkit.runner import run_experiment

app = typer.Typer(help="Neural Bending Toolkit command-line interface.")


GEOPOLITICAL_EXPERIMENT = "geopolitical-bend"
geopolitical_app = typer.Typer(help="Geopolitical Bend experiment commands.")
app.add_typer(geopolitical_app, name="geopolitical")


def _load_registry() -> ExperimentRegistry:
    registry = ExperimentRegistry()
    registry.discover()
    return registry


@app.command("list")
def list_experiments() -> None:
    """List available experiments."""
    registry = _load_registry()
    for name in registry.list_experiments():
        typer.echo(name)


@app.command("describe")
def describe_experiment(experiment: str) -> None:
    """Describe an experiment and show its config schema."""
    registry = _load_registry()
    exp_cls = registry.get(experiment)
    typer.echo(json.dumps(exp_cls.describe(), indent=2))


@app.command("run")
def run(
    experiment: str,
    config: Annotated[
        Path,
        typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    ],
) -> None:
    """Run an experiment with a YAML configuration file."""
    registry = _load_registry()
    run_dir = run_experiment(experiment, config, registry)
    typer.echo(f"Run completed: {run_dir}")


@app.command("report")
def report(run_dir: Path, output: str = "report.md") -> None:
    """Generate a markdown report with analysis and artifact citations."""
    report_path = generate_markdown_report(run_dir=run_dir, output_name=output)
    typer.echo(f"Report generated: {report_path}")


@geopolitical_app.command("describe")
def geopolitical_describe() -> None:
    """Describe the Geopolitical Bend experiment schema."""
    registry = _load_registry()
    try:
        exp_cls = registry.get(GEOPOLITICAL_EXPERIMENT)
    except KeyError as err:
        raise typer.BadParameter(
            "Geopolitical experiment is not registered. "
            "Check entry points and installation."
        ) from err
    typer.echo(json.dumps(exp_cls.describe(), indent=2))


@geopolitical_app.command("run")
def geopolitical_run(
    config: Annotated[
        Path,
        typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    ],
) -> None:
    """Run Geopolitical Bend with configuration file."""
    if config.suffix.lower() not in {".yaml", ".yml"}:
        raise typer.BadParameter("Config must be a YAML file (.yaml/.yml).")

    registry = _load_registry()
    try:
        run_dir = run_experiment(GEOPOLITICAL_EXPERIMENT, config, registry)
    except KeyError as err:
        raise typer.BadParameter(
            "Geopolitical experiment is not registered. "
            "Run `nbt list` to inspect available experiments."
        ) from err
    except Exception as err:
        message = f"Failed to run geopolitical experiment: {err}"
        raise typer.BadParameter(message) from err

    typer.echo(f"Geopolitical run completed: {run_dir}")


@geopolitical_app.command("report")
def geopolitical_report(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    format: Annotated[
        str, typer.Option("--format", help="markdown or html")
    ] = "markdown",
) -> None:
    """Generate a geopolitical report from a run directory."""
    if format not in {"markdown", "html"}:
        raise typer.BadParameter("--format must be 'markdown' or 'html'.")

    try:
        out_path = generate_geopolitical_report(run_dir, format=format)
    except Exception as err:
        raise typer.BadParameter(f"Failed to generate report: {err}") from err

    typer.echo(f"Geopolitical report generated: {out_path}")


@geopolitical_app.command("compare")
def geopolitical_compare(
    run_dirs: Annotated[list[Path], typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Compare two or more geopolitical runs."""
    if len(run_dirs) < 2:
        raise typer.BadParameter("Provide at least two run directories.")

    try:
        csv_path, md_path = compare_geopolitical_runs(run_dirs)
    except Exception as err:
        raise typer.BadParameter(f"Failed to compare runs: {err}") from err

    typer.echo(f"Geopolitical comparison CSV: {csv_path}")
    typer.echo(f"Geopolitical comparison summary: {md_path}")
