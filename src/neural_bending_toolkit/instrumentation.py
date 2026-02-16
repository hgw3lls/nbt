"""Instrumentation primitives for logs, metrics, artifacts, and integrations."""

from __future__ import annotations

import importlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class StructuredLogger:
    """Dual logger that emits human-readable and JSON log events."""

    def __init__(self, text_log_path: Path, json_log_path: Path) -> None:
        self.text_log_path = text_log_path
        self.json_log_path = json_log_path

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def log(self, level: str, event: str, **payload: Any) -> None:
        timestamp = self._now()
        record = {
            "timestamp": timestamp,
            "level": level.upper(),
            "event": event,
            "payload": payload,
        }

        message = payload.get("message", "")
        with self.text_log_path.open("a", encoding="utf-8") as text_file:
            text_file.write(f"{timestamp} [{level.upper()}] {event} {message}\n")

        with self.json_log_path.open("a", encoding="utf-8") as json_file:
            json_file.write(json.dumps(record) + "\n")


class MetricsLogger:
    """Writes structured metrics records as JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def log(
        self,
        *,
        step: int,
        metric_name: str,
        value: float | int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "timestamp": self._now(),
            "step": step,
            "metric_name": metric_name,
            "value": value,
            "metadata": metadata or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class ArtifactSaver:
    """Helpers for saving common artifact formats under run artifacts directory."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, filename: str) -> Path:
        path = self.artifacts_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_text(self, filename: str, text: str) -> Path:
        path = self._path(filename)
        path.write_text(text, encoding="utf-8")
        return path

    def save_numpy(self, filename: str, array: np.ndarray) -> Path:
        path = self._path(filename)
        np.save(path, array)
        return path

    def save_image(self, filename: str, image: Any) -> Path:
        path = self._path(filename)

        if isinstance(image, np.ndarray):
            if importlib.util.find_spec("PIL") is None:
                raise RuntimeError("Pillow is required to save numpy arrays as images.")
            pil_image_module = importlib.import_module("PIL.Image")
            pil_image = pil_image_module.fromarray(image)
            pil_image.save(path)
            return path

        if hasattr(image, "save"):
            image.save(path)
            return path

        raise TypeError(
            "Unsupported image type. Provide a numpy array or PIL-like image."
        )

    def save_torch_tensor(self, filename: str, tensor: Any) -> Path:
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("PyTorch is not installed.")
        torch = importlib.import_module("torch")
        path = self._path(filename)
        torch.save(tensor, path)
        return path


class WandbLogger:
    """Optional Weights & Biases integration behind a feature flag."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        run_name: str,
        config: dict[str, Any],
    ) -> None:
        self.enabled = enabled
        self._run = None
        if not enabled:
            return
        if importlib.util.find_spec("wandb") is None:
            return

        wandb = importlib.import_module("wandb")
        self._run = wandb.init(project=project, name=run_name, config=config)

    def log_metric(
        self,
        *,
        step: int,
        metric_name: str,
        value: float | int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._run:
            return
        payload = {metric_name: value, **(metadata or {})}
        self._run.log(payload, step=step)

    def finish(self) -> None:
        if self._run:
            self._run.finish()
