"""Bend primitives for intervention experiments."""

from neural_bending_toolkit.bends.base import BendMetadata, BendPrimitive
from neural_bending_toolkit.bends.primitives import (
    AttentionConflictInjector,
    AttentionHeadScaler,
    AttractorSeeder,
    EmbeddingBlend,
    JusticeReweighter,
    LowProbSampler,
    NormStatPerturber,
    ResidualNoiseInjector,
)
from neural_bending_toolkit.bends.v2 import (
    ActuatorSpec,
    BendPlan,
    BendSpec,
    ScheduleSpec,
    SiteSpec,
    TraceSpec,
    bend_localizability_label,
)
from neural_bending_toolkit.bends.v2_diffusion import (
    compile_diffusion_cross_attention_hook,
)

__all__ = [
    "ActuatorSpec",
    "BendPlan",
    "BendSpec",
    "bend_localizability_label",
    "compile_diffusion_cross_attention_hook",
    "AttractorSeeder",
    "AttentionConflictInjector",
    "AttentionHeadScaler",
    "BendMetadata",
    "BendPrimitive",
    "EmbeddingBlend",
    "ScheduleSpec",
    "SiteSpec",
    "TraceSpec",
    "JusticeReweighter",
    "LowProbSampler",
    "NormStatPerturber",
    "ResidualNoiseInjector",
]
