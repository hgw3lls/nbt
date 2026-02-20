"""Experiment abstractions and run context."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from neural_bending_toolkit.figures.specs import FigureSpec, save_figure_spec
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

    @staticmethod
    def _has_metric_name(run_dir: Path, metric_name: str) -> bool:
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            return False

        target = metric_name.strip().lower()
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            name = str(record.get("metric_name", "")).strip().lower()
            if name == target:
                return True
        return False

    def recommended_figure_specs(self, run_dir: Path) -> list[dict[str, Any]]:
        """Return recommended figure specs for the run (override in subclasses)."""
        run_dir = Path(run_dir)
        analysis_dir = run_dir / "analysis"
        derived = {}
        classification = {}
        derived_path = analysis_dir / "derived_metrics.json"
        classification_path = analysis_dir / "bend_classification.json"
        if derived_path.exists():
            derived = json.loads(derived_path.read_text(encoding="utf-8"))
        if classification_path.exists():
            classification = json.loads(
                classification_path.read_text(encoding="utf-8")
            )

        specs: list[dict[str, Any]] = [
            {
                "figure_id": f"{run_dir.name}_divergence",
                "title": "Divergence Overview",
                "input_run_dirs": [str(run_dir)],
                "plot_type": "divergence_bar_chart",
                "inputs": {},
                "output_format": {"png": True, "pdf": True},
                "caption_template_variables": {
                    "limit": "distribution drift",
                    "threshold": "divergence increase",
                },
            }
        ]

        bend_tag = str(classification.get("bend_tag", ""))
        has_attention_entropy = self._has_metric_name(run_dir, "attention_entropy")
        if bend_tag == "disruptive" or has_attention_entropy:
            specs.append(
                {
                    "figure_id": f"{run_dir.name}_attention_entropy",
                    "title": "Attention Entropy Timeseries",
                    "input_run_dirs": [str(run_dir)],
                    "plot_type": "attention_entropy_timeseries",
                    "inputs": {},
                    "output_format": {"png": True, "pdf": True},
                    "caption_template_variables": {
                        "limit": "entropy escalation",
                        "threshold": "abrupt coherence break",
                    },
                }
            )

        if isinstance(derived.get("attractor_density_delta"), (int, float)):
            specs.append(
                {
                    "figure_id": f"{run_dir.name}_attractor_density",
                    "title": "Attractor Density Comparison",
                    "input_run_dirs": [str(run_dir)],
                    "plot_type": "attractor_density_comparison",
                    "inputs": {},
                    "output_format": {"png": True, "pdf": True},
                    "caption_template_variables": {
                        "limit": "token attractor saturation",
                        "threshold": "density phase shift",
                    },
                }
            )

        if list((run_dir / "artifacts").glob("**/*.png")):
            specs.append(
                {
                    "figure_id": f"{run_dir.name}_artifact_montage",
                    "title": "Artifact Montage",
                    "input_run_dirs": [str(run_dir)],
                    "plot_type": "montage_grid",
                    "inputs": {},
                    "output_format": {"png": True, "pdf": True},
                    "caption_template_variables": {
                        "limit": "visible sample drift",
                        "threshold": "qualitative regime change",
                    },
                }
            )
        return specs

    def emit_figure_specs(self, run_dir: Path) -> list[Path]:
        """Write recommended figure specs under run_dir/figure_specs."""
        run_dir = Path(run_dir)
        specs_dir = run_dir / "figure_specs"
        specs_dir.mkdir(parents=True, exist_ok=True)

        emitted: list[Path] = []
        for payload in self.recommended_figure_specs(run_dir):
            spec = FigureSpec.model_validate(payload)
            out_path = specs_dir / f"{spec.figure_id}.yaml"
            save_figure_spec(spec, out_path)
            emitted.append(out_path)
        return emitted

    @abstractmethod
    def run(self, context: RunContext) -> None:
        """Run the experiment using the given context."""
