"""CLI entrypoint for neural_bending_toolkit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

from neural_bending_toolkit.analysis.geopolitical_report import (
    compare_geopolitical_runs,
    generate_geopolitical_report,
)
from neural_bending_toolkit.analysis.report import generate_markdown_report
from neural_bending_toolkit.figures import (
    build_figure_from_spec,
    build_figures_from_run,
)
from neural_bending_toolkit.registry import ExperimentRegistry
from neural_bending_toolkit.runner import run_experiment

app = typer.Typer(help="Neural Bending Toolkit command-line interface.")
init_app = typer.Typer(help="Project initialization commands.")
figure_app = typer.Typer(help="Figure build commands.")
app.add_typer(init_app, name="init")
app.add_typer(figure_app, name="figure")


GEOPOLITICAL_EXPERIMENT = "geopolitical-bend"
geopolitical_app = typer.Typer(help="Geopolitical Bend experiment commands.")
app.add_typer(geopolitical_app, name="geopolitical")


def _load_registry() -> ExperimentRegistry:
    registry = ExperimentRegistry()
    registry.discover()
    return registry


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@app.command("docs")
def docs() -> None:
    """Print core documentation and template locations."""
    root = _repo_root()
    docs_dir = root / "docs"
    templates_dir = root / "templates"

    lines = [
        "Neural Bending Toolkit documentation guide",
        "",
        f"Repo root: {root}",
        f"Docs directory: {docs_dir}",
        f"Templates directory: {templates_dir}",
        "",
        "Start with:",
        f"- {docs_dir / 'OVERVIEW.md'}",
        f"- {docs_dir / 'RUNS_AND_ARTIFACTS.md'}",
        f"- {docs_dir / 'FIGURES_AND_REPORTS.md'}",
        f"- {docs_dir / 'GEOPOLITICAL_BEND.md'}",
        "",
        "Dissertation setup:",
        "- Run `nbt init dissertation` to create dissertation/ scaffolding.",
        "- Use templates/ for theory memos, captions, and methods appendix drafts.",
    ]
    typer.echo("\n".join(lines))


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


@init_app.command("dissertation")
def init_dissertation(
    target: Annotated[Path, typer.Option("--path", help="Output root path")] = Path(
        "."
    ),
) -> None:
    """Initialize dissertation research-creation folder scaffold."""
    root = target.resolve()
    dissertation_dir = root / "dissertation"
    subdirs = ["figures", "tables", "memos", "exports"]

    dissertation_dir.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        (dissertation_dir / subdir).mkdir(parents=True, exist_ok=True)

    readme = dissertation_dir / "README.md"
    created_utc = datetime.now(timezone.utc).isoformat()
    readme.write_text(
        "\n".join(
            [
                "# Dissertation Workspace",
                "",
                f"Created (UTC): {created_utc}",
                "",
                "## Workflow",
                "1. Run experiments in `runs/` using NBT CLI.",
                "2. Generate reports with `nbt report` or `nbt geopolitical report`.",
                (
                    "3. Copy or curate selected artifacts into "
                    "`dissertation/figures` and `dissertation/exports`."
                ),
                (
                    "4. Draft interpretive notes in `dissertation/memos` "
                    "using templates/theory_memo.md."
                ),
                "5. Build chapter tables in `dissertation/tables`.",
                "",
                "## Suggested command sequence",
                "- `nbt init dissertation`",
                "- `nbt list`",
                "- `nbt run <experiment> --config <config.yaml>`",
                "- `nbt report runs/<timestamp>_<experiment>`",
                "- `nbt docs`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    typer.echo(f"Initialized dissertation workspace: {dissertation_dir}")


@figure_app.command("build")
def figure_build(
    spec: Annotated[Path, typer.Option("--spec", exists=True, dir_okay=False)],
) -> None:
    """Build one dissertation figure from a YAML spec."""
    if spec.suffix.lower() not in {".yaml", ".yml"}:
        raise typer.BadParameter("Spec must be a YAML file (.yaml/.yml).")

    try:
        output_dir = build_figure_from_spec(spec)
    except Exception as err:
        raise typer.BadParameter(f"Failed to build figure from spec: {err}") from err

    typer.echo(f"Figure built: {output_dir}")


@figure_app.command("build-from-run")
def figure_build_from_run(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Build all figure specs emitted in run_dir/figure_specs."""
    try:
        outputs = build_figures_from_run(run_dir)
    except Exception as err:
        raise typer.BadParameter(f"Failed to build figures from run: {err}") from err

    typer.echo(f"Built {len(outputs)} figure(s):")
    for output in outputs:
        typer.echo(f"- {output}")
