"""Experiment abstractions and run context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from neural_bending_toolkit.instrumentation import (
    ArtifactSaver,
    MetricsLogger,
    StructuredLogger,
    WandbLogger,
)


class ExperimentSettings(BaseSettings):
    """Base class for experiment configuration schema."""

    model_config = SettingsConfigDict(extra="forbid")


class RunContext:
    """Context object shared with an experiment during execution."""

    def __init__(
        self,
        run_dir: Path,
        *,
        wandb_enabled: bool = False,
        wandb_project: str = "neural-bending-toolkit",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.artifacts_dir = run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.logger = StructuredLogger(
            text_log_path=run_dir / "events.log",
            json_log_path=run_dir / "events.jsonl",
        )
        self.metrics = MetricsLogger(path=run_dir / "metrics.jsonl")
        self.artifacts_saver = ArtifactSaver(self.artifacts_dir)
        self.wandb = WandbLogger(
            enabled=wandb_enabled,
            project=wandb_project,
            run_name=run_name or run_dir.name,
            config=config or {},
        )

    def log_event(self, message: str, level: str = "info", **payload: Any) -> None:
        self.logger.log(level=level, event="event", message=message, **payload)

    def log_metric(
        self,
        *,
        step: int,
        metric_name: str,
        value: float | int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.metrics.log(
            step=step,
            metric_name=metric_name,
            value=value,
            metadata=metadata,
        )
        self.wandb.log_metric(
            step=step,
            metric_name=metric_name,
            value=value,
            metadata=metadata,
        )

    def save_text_artifact(self, filename: str, text: str) -> Path:
        path = self.artifacts_saver.save_text(filename, text)
        self.log_event(f"Saved text artifact: {path.name}", artifact=str(path))
        return path

    def save_image_artifact(self, filename: str, image: Any) -> Path:
        path = self.artifacts_saver.save_image(filename, image)
        self.log_event(f"Saved image artifact: {path.name}", artifact=str(path))
        return path

    def save_numpy_artifact(self, filename: str, array: Any) -> Path:
        path = self.artifacts_saver.save_numpy(filename, array)
        self.log_event(f"Saved numpy artifact: {path.name}", artifact=str(path))
        return path

    def save_torch_artifact(self, filename: str, tensor: Any) -> Path:
        path = self.artifacts_saver.save_torch_tensor(filename, tensor)
        self.log_event(f"Saved torch artifact: {path.name}", artifact=str(path))
        return path

    def pre_intervention_snapshot(self, name: str, data: dict[str, Any]) -> None:
        self.logger.log(
            level="info",
            event="pre_intervention_snapshot",
            name=name,
            snapshot=data,
        )

    def post_intervention_snapshot(self, name: str, data: dict[str, Any]) -> None:
        self.logger.log(
            level="info",
            event="post_intervention_snapshot",
            name=name,
            snapshot=data,
        )

    def close(self) -> None:
        self.wandb.finish()


class Experiment(ABC):
    """Base class for toolkit experiments."""

    name: ClassVar[str]
    config_model: ClassVar[type[BaseModel]] = ExperimentSettings

    def __init__(self, config: BaseModel) -> None:
        self.config = config

    @classmethod
    def experiment_name(cls) -> str:
        return cls.name

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {
            "name": cls.experiment_name(),
            "description": (cls.__doc__ or "").strip(),
            "config_schema": cls.config_model.model_json_schema(),
        }

    def artifacts(self, run_dir: Path) -> list[Path]:
        artifacts_dir = run_dir / "artifacts"
        if not artifacts_dir.exists():
            return []
        return sorted(path for path in artifacts_dir.glob("**/*") if path.is_file())

    def recommended_figure_specs(self, run_dir: Path) -> list[dict[str, Any]]:
        """Return optional figure spec payloads to emit into run_dir/figure_specs."""
        return []

    def emit_figure_specs(self, run_dir: Path) -> list[Path]:
        """Write recommended figure specs into run_dir/figure_specs."""
        specs = self.recommended_figure_specs(run_dir)
        if not specs:
            return []

        specs_dir = run_dir / "figure_specs"
        specs_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for idx, payload in enumerate(specs, start=1):
            if not isinstance(payload, dict):
                continue
            figure_id = str(payload.get("figure_id", f"figure_{idx}"))
            safe_id = figure_id.replace("/", "_").replace(" ", "_")
            out_path = specs_dir / f"{safe_id}.yaml"
            with out_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)
            written.append(out_path)

        return written

    @abstractmethod
    def run(self, context: RunContext) -> None:
        """Run the experiment using the given context."""
