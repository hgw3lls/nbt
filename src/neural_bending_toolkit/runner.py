"""Experiment execution helpers."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from neural_bending_toolkit.analysis.bend_classifier import write_bend_classification
from neural_bending_toolkit.analysis.derived_metrics import write_derived_metrics
from neural_bending_toolkit.analysis.theory_memo_generator import build_theory_memo
from neural_bending_toolkit.config import validate_config_dict
from neural_bending_toolkit.experiment import RunContext
from neural_bending_toolkit.registry import ExperimentRegistry

RUNS_ROOT = Path("runs")


def build_run_dir(experiment_name: str, runs_root: Path = RUNS_ROOT) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_root / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    return run_dir


def run_experiment(
    experiment_name: str,
    config_path: Path,
    registry: ExperimentRegistry,
) -> Path:
    experiment_cls = registry.get(experiment_name)

    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    auto_memo = (
        bool(raw_config.pop("auto_memo", False))
        if isinstance(raw_config, dict)
        else False
    )
    config = validate_config_dict(raw_config, experiment_cls.config_model)

    run_dir = build_run_dir(experiment_name)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=True)

    context = RunContext(
        run_dir,
        wandb_enabled=os.getenv("NBT_ENABLE_WANDB", "false").lower() == "true",
        run_name=run_dir.name,
        config=config.model_dump(),
    )
    experiment = experiment_cls(config)
    try:
        experiment.run(context)
    finally:
        context.close()

    if auto_memo:
        write_derived_metrics(run_dir)
        write_bend_classification(run_dir)
        build_theory_memo(run_dir)

    experiment.emit_figure_specs(run_dir)
    return run_dir
