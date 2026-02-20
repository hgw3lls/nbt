"""Bend v2 specification models."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SiteSpec(BaseModel):
    """Where in model execution an intervention applies."""

    kind: Literal[
        "diffusion.cross_attention",
        "diffusion.norm",
        "llm.attention",
        "gan.layer",
        "audio.attention",
    ] = "diffusion.cross_attention"
    layer_regex: str | None = None
    layer_names: list[str] | None = None
    allow_all_layers: bool = False
    head_indices: list[int] | None = None
    token_indices: list[int] | None = None
    timestep_start: int | None = None
    timestep_end: int | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_localization(self) -> SiteSpec:
        has_layer_selector = bool(self.layer_regex) or bool(self.layer_names)
        if not has_layer_selector and not self.allow_all_layers:
            raise ValueError(
                "SiteSpec requires layer_regex or layer_names, "
                "or allow_all_layers=True"
            )

        if self.timestep_start is not None and self.timestep_end is not None:
            if self.timestep_start > self.timestep_end:
                raise ValueError("timestep_start must be <= timestep_end")

        if self.head_indices is not None:
            if any(idx < 0 for idx in self.head_indices):
                raise ValueError("head_indices must contain non-negative integers")

        return self


class ScheduleSpec(BaseModel):
    """When and how strongly an intervention is applied."""

    mode: Literal["constant", "ramp", "pulse", "window"] = "constant"
    strength: float
    strength_start: float | None = None
    strength_end: float | None = None
    period: int | None = None
    duty: float | None = None
    condition: str | None = None

    @model_validator(mode="after")
    def validate_mode_fields(self) -> ScheduleSpec:
        if not math.isfinite(self.strength):
            raise ValueError("strength must be finite")

        if self.mode == "ramp":
            if self.strength_start is None or self.strength_end is None:
                raise ValueError(
                    "strength_start and strength_end are required when mode='ramp'"
                )
            if not math.isfinite(self.strength_start) or not math.isfinite(
                self.strength_end
            ):
                raise ValueError("strength_start and strength_end must be finite")

        if self.mode == "pulse":
            if self.period is None:
                raise ValueError("period is required when mode='pulse'")
            if self.duty is None:
                raise ValueError("duty is required when mode='pulse'")
            if self.period <= 0:
                raise ValueError("period must be > 0")
            if not math.isfinite(self.duty):
                raise ValueError("duty must be finite")
            if not 0.0 < self.duty <= 1.0:
                raise ValueError("duty must be in (0.0, 1.0]")

        return self


class ActuatorSpec(BaseModel):
    """What perturbation is applied at the selected site."""

    type: Literal[
        "attention_head_gate",
        "attention_probs_temperature",
        "qk_rotate",
        "kv_noise",
        "embedding_project",
        "norm_gain_drift",
        "norm_bias_shift",
        "norm_stat_clamp",
        "activation_noise",
        "noop",
    ]
    params: dict[str, Any] = Field(default_factory=dict)


class TraceSpec(BaseModel):
    """Trace payload options for bend execution."""

    metrics: list[
        Literal[
            "attention_entropy",
            "attention_topk_mass",
            "latent_delta_l2",
            "activation_norm",
            "norm_stats",
            "norm_output_mean",
            "norm_output_var",
            "activation_snr",
        ]
    ]
    sample_every: int = Field(default=1, ge=1)
    save_raw: bool = False


class BendSpec(BaseModel):
    """Top-level bend declaration."""

    name: str
    site: SiteSpec
    actuator: ActuatorSpec
    schedule: ScheduleSpec
    trace: TraceSpec | None = None
    tags: list[str] = Field(default_factory=list)
    safety: dict[str, Any] | None = None


class BendPlan(BaseModel):
    """A serializable collection of bend specifications."""

    bends: list[BendSpec]


def bend_localizability_label(bend: BendSpec) -> str:
    """Build a short human-readable location label for a bend site."""

    if bend.site.allow_all_layers:
        layer_text = "all"
    elif bend.site.layer_regex:
        layer_text = bend.site.layer_regex
    elif bend.site.layer_names:
        layer_text = ",".join(bend.site.layer_names)
    else:
        layer_text = "unspecified"

    heads = bend.site.head_indices
    head_text = "all" if heads is None else "[" + ",".join(str(i) for i in heads) + "]"

    step_start = bend.site.timestep_start
    step_end = bend.site.timestep_end
    if step_start is None and step_end is None:
        steps_text = "all"
    elif step_start is None:
        steps_text = f"..{step_end}"
    elif step_end is None:
        steps_text = f"{step_start}.."
    else:
        steps_text = f"{step_start}..{step_end}"

    return (
        f"{bend.site.kind}: layer={layer_text} "
        f"head={head_text} steps={steps_text}"
    )
