"""CLI entrypoint for neural_bending_toolkit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from neural_bending_toolkit.analysis.bend_classifier import write_bend_classification
from neural_bending_toolkit.analysis.derived_metrics import write_derived_metrics
from neural_bending_toolkit.analysis.geopolitical_report import (
    compare_geopolitical_runs,
    generate_geopolitical_report,
)
from neural_bending_toolkit.analysis.report import (
    generate_html_report,
    generate_markdown_report,
)
from neural_bending_toolkit.analysis.theory_memo_generator import (
    build_theory_memo,
    build_theory_memos_for_runs,
)
from neural_bending_toolkit.console.runtime import node_specs
from neural_bending_toolkit.console.schema import validate_patch_graph
from neural_bending_toolkit.registry import ExperimentRegistry
from neural_bending_toolkit.runner import run_experiment

app = typer.Typer(help="Neural Bending Toolkit command-line interface.")


GEOPOLITICAL_EXPERIMENT = "geopolitical-bend"
geopolitical_app = typer.Typer(help="Geopolitical Bend experiment commands.")
app.add_typer(geopolitical_app, name="geopolitical")
analyze_app = typer.Typer(help="Post-run derived metric analysis and classification.")
app.add_typer(analyze_app, name="analyze")
memo_app = typer.Typer(help="Build theory memos from run outputs.")
app.add_typer(memo_app, name="memo")
console_app = typer.Typer(help="Interactive console commands.")
app.add_typer(console_app, name="console")


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


@app.command("curate")
def curate(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Curate memo, analysis, reports, and top figures into one export folder."""
    run_dir = Path(run_dir)
    markdown_report = generate_markdown_report(run_dir, output_name="report.md")
    html_report = generate_html_report(run_dir, output_name="report.html")

    analysis_dir = run_dir / "analysis"
    outputs = [
        run_dir / "theory_memo.md",
        analysis_dir / "bend_classification.json",
        analysis_dir / "derived_metrics.json",
        markdown_report,
        html_report,
    ]

    figures = sorted(
        list((analysis_dir).glob("**/*.png"))
        + list((run_dir / "artifacts").glob("**/*.png"))
    )
    top_figures = figures[:4]

    curated_dir = run_dir / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)
    caption_lines = ["# Curated Figure Captions", ""]

    for path in outputs + top_figures:
        if not path.exists():
            continue
        target = curated_dir / path.name
        target.write_bytes(path.read_bytes())
        typer.echo(f"Curated: {target}")
        if path in top_figures:
            caption_lines.append(f"- `{path.name}`: auto-curated top figure reference")

    (curated_dir / "caption.md").write_text(
        "\n".join(caption_lines) + "\n", encoding="utf-8"
    )
    typer.echo(f"Curated bundle created: {curated_dir}")


@analyze_app.command("derive")
def analyze_derive(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Derive standardized metrics from run outputs."""
    out_path = write_derived_metrics(run_dir)
    typer.echo(f"Derived metrics written: {out_path}")


@analyze_app.command("classify")
def analyze_classify(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Classify run bend type from derived metrics."""
    out_path = write_bend_classification(run_dir)
    typer.echo(f"Bend classification written: {out_path}")


@analyze_app.command("all")
def analyze_all(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Run derive + classify in sequence for a run directory."""
    derived_path = write_derived_metrics(run_dir)
    class_path = write_bend_classification(run_dir)
    typer.echo(f"Derived metrics written: {derived_path}")
    typer.echo(f"Bend classification written: {class_path}")


@memo_app.command("build")
def memo_build(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Build theory memo for one run."""
    out_path = build_theory_memo(run_dir)
    typer.echo(f"Theory memo generated: {out_path}")


@memo_app.command("build-all")
def memo_build_all(
    runs_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Build theory memos for all run directories under runs_root."""
    outputs = build_theory_memos_for_runs(runs_root)
    for path in outputs:
        typer.echo(f"Theory memo generated: {path}")
    if not outputs:
        typer.echo("No run directories found with config.yaml")


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


@console_app.command("serve")
def console_serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the console websocket server."""
    try:
        import uvicorn
    except ImportError as err:
        raise typer.BadParameter(
            "uvicorn is required for `nbt console serve`. Install with `pip install uvicorn`."
        ) from err
    from neural_bending_toolkit.console.server import app as console_server_app

    if console_server_app is None:
        raise typer.BadParameter(
            "fastapi is required for `nbt console serve`. Install with `pip install fastapi`."
        )

    uvicorn.run(console_server_app, host=host, port=port)


@console_app.command("validate")
def console_validate(
    patch: Annotated[list[Path], typer.Option("--patch", exists=True, dir_okay=False)],
) -> None:
    """Validate one or more patch JSON files."""
    for patch_file in patch:
        payload = json.loads(patch_file.read_text(encoding="utf-8"))
        validate_patch_graph(payload, node_specs=node_specs())
        typer.echo(f"valid patch: {patch_file}")


@console_app.command("init")
def console_init(output: Path = Path("patches/starter_patch.json")) -> None:
    """Write a starter console patch."""
    output.parent.mkdir(parents=True, exist_ok=True)
    starter_patch = {
        "nodes": [
            {
                "id": "prompt_1",
                "type": "PromptSourceNode",
                "params": {"text": "neural bending console"},
                "ui": {"x": 80, "y": 120},
                "enabled": True,
            },
            {
                "id": "gen_1",
                "type": "DummyTextGenNode",
                "params": {},
                "ui": {"x": 340, "y": 120},
                "enabled": True,
            },
            {
                "id": "metrics_1",
                "type": "MetricProbeNode",
                "params": {},
                "ui": {"x": 600, "y": 120},
                "enabled": True,
            },
        ],
        "edges": [
            {
                "id": "e_prompt_gen",
                "from_node": "prompt_1",
                "from_port": "text",
                "to_node": "gen_1",
                "to_port": "prompt",
            },
            {
                "id": "e_gen_metrics",
                "from_node": "gen_1",
                "from_port": "text",
                "to_node": "metrics_1",
                "to_port": "text",
            },
        ],
        "globals": {"tick_rate": 30},
    }
    output.write_text(json.dumps(starter_patch, indent=2), encoding="utf-8")
    typer.echo(f"starter patch written: {output}")


@console_app.command("patches")
def console_patches() -> None:
    """List available starter patch files under patches/."""
    patches_dir = Path("patches")
    if not patches_dir.exists():
        typer.echo("No patches directory found.")
        return

    patch_files = sorted(patches_dir.glob("*.json"))
    if not patch_files:
        typer.echo("No patch files found.")
        return

    for patch_file in patch_files:
        description = "(no description)"
        try:
            payload = json.loads(patch_file.read_text(encoding="utf-8"))
            notes = payload.get("globals", {}).get("notes")
            if isinstance(notes, str) and notes.strip():
                description = notes.strip().split(".")[0] + "."
        except Exception:
            description = "(unreadable patch metadata)"
        typer.echo(f"{patch_file}: {description}")
