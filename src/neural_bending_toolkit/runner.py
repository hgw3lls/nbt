"""Experiment execution helpers."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from neural_bending_toolkit.config import load_and_validate_config
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
    config = load_and_validate_config(config_path, experiment_cls.config_model)

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
    return run_dir
