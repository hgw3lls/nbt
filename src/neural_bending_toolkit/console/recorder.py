"""Console run recorder utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ConsoleRecorder:
    """Writes console runtime artifacts using toolkit-like run folders."""

    def __init__(self, runs_root: Path = Path("runs")) -> None:
        self.runs_root = runs_root
        self.run_dir: Path | None = None
        self.metrics_path: Path | None = None
        self.outputs_path: Path | None = None

    def start(self, patch: dict[str, Any], config_snapshot: dict[str, Any]) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = self.runs_root / f"console_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "figure_specs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        (self.run_dir / "patch.json").write_text(
            json.dumps(patch, indent=2), encoding="utf-8"
        )
        (self.run_dir / "config_snapshot.json").write_text(
            json.dumps(config_snapshot, indent=2), encoding="utf-8"
        )
        (self.run_dir / "theory_memo.md").write_text("# Theory memo\n", encoding="utf-8")

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.outputs_path = self.run_dir / "outputs" / "text.txt"
        return self.run_dir

    def record_metric(self, metric: dict[str, Any]) -> None:
        if not self.metrics_path:
            return
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metric) + "\n")

    def record_text(self, text_chunk: str) -> None:
        if not self.outputs_path:
            return
        with self.outputs_path.open("a", encoding="utf-8") as f:
            f.write(text_chunk)

    def save_take(self, label: str | None = None) -> Path | None:
        if not self.run_dir:
            return None
        if label:
            take_file = self.run_dir / f"take_{label}.txt"
            take_file.write_text(
                f"take captured: {datetime.now(timezone.utc).isoformat()}\n",
                encoding="utf-8",
            )
        return self.run_dir
