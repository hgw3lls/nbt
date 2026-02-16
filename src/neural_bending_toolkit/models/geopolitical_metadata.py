"""Geopolitical model registry helper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _registry_path() -> Path:
    return Path(__file__).with_name("geopolitical_models.json")


def get_geopolitical_metadata(model_id: str) -> dict[str, Any] | None:
    """Lookup model metadata in the geopolitical model registry."""
    path = _registry_path()
    if not path.exists():
        return None
    records = json.loads(path.read_text(encoding="utf-8"))
    for record in records:
        if record.get("model_identifier") == model_id:
            return record
    return None
