"""Figure specification models and parsing utilities."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PlotType(str, Enum):
    EMBEDDING_SIMILARITY_HEATMAP = "embedding_similarity_heatmap"
    EMBEDDING_UMAP_SCATTER = "embedding_umap_scatter"
    ATTENTION_ENTROPY_TIMESERIES = "attention_entropy_timeseries"
    DIVERGENCE_BAR_CHART = "divergence_bar_chart"
    REFUSAL_RATE_TABLE_TO_FIGURE = "refusal_rate_table_to_figure"
    CAUSAL_FRAMING_BAR_CHART = "causal_framing_bar_chart"
    ATTRACTOR_DENSITY_COMPARISON = "attractor_density_comparison"
    MONTAGE_GRID = "montage_grid"


class OutputFormat(BaseModel):
    """Output format settings for figure exports."""

    model_config = ConfigDict(extra="forbid")

    png: bool = True
    pdf: bool = True

    @model_validator(mode="after")
    def validate_at_least_one(self) -> OutputFormat:
        if not (self.png or self.pdf):
            raise ValueError("At least one output format must be enabled.")
        return self


class FigureSpec(BaseModel):
    """Strict schema for figure specifications."""

    model_config = ConfigDict(extra="forbid")

    figure_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    input_run_dirs: list[str] = Field(min_length=1)
    plot_type: PlotType
    inputs: dict[str, Any] = Field(default_factory=dict)
    output_format: OutputFormat = Field(default_factory=OutputFormat)
    caption_template_variables: dict[str, str] = Field(default_factory=dict)

    @field_validator("figure_id")
    @classmethod
    def validate_figure_id(cls, value: str) -> str:
        if any(ch in value for ch in "\\/ "):
            raise ValueError("figure_id must not contain slashes or spaces")
        return value


def load_figure_spec(path: Path) -> FigureSpec:
    """Load and validate a figure specification YAML file."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError("Figure spec root must be a YAML mapping/object.")

    return FigureSpec.model_validate(payload)


def save_figure_spec(spec: FigureSpec, path: Path) -> None:
    """Save a FigureSpec as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec.model_dump(mode="python"), handle, sort_keys=False)
