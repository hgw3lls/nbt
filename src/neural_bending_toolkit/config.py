"""Configuration loading and validation for experiments."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError


class ConfigValidationError(ValueError):
    """Raised when experiment config does not validate."""


def validate_config_dict(raw: object, model: type[BaseModel]) -> BaseModel:
    """Validate a pre-loaded config mapping against a pydantic model."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigValidationError("Config file root must be a mapping/object.")

    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError(str(exc)) from exc


def load_and_validate_config(path: Path, model: type[BaseModel]) -> BaseModel:
    """Load YAML config and validate with the provided pydantic model."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return validate_config_dict(raw, model)
