"""Multimodal alignment bends (stubbed for future expansion)."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from .base import BendResult, NeuralBend, register_bend


class _AlignmentFrictionWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, scale: float) -> None:
        super().__init__()
        self.base_model = base_model
        self.scale = scale

    def forward(self, *args, **kwargs):  # type: ignore[override]
        # Minimal example: if model exposes a projection matrix between modalities,
        # soften it so the two channels have to renegotiate alignment.
        if hasattr(self.base_model, "visual_projection"):
            projection = getattr(self.base_model, "visual_projection")
            with torch.no_grad():
                projection.weight.mul_(self.scale)
        return self.base_model(*args, **kwargs)


class CrossModalFrictionBend(NeuralBend):
    """Introduce friction in multimodal alignment layers.

    Technical: scales cross-modal projection weights, encouraging drift between
    text and image/audio streams.
    Media-theoretical: acknowledges that alignment is contested; friction makes
    the seams audible when one modality overrules another.
    """

    def __init__(self) -> None:
        super().__init__(
            name="multimodal_alignment_friction",
            domain="multimodal",
            category="disruptive",
            description="Loosen cross-modal projections so alignment becomes a visible negotiation.",
            technical_notes="Scales visual_projection weights if present; otherwise acts as pass-through.",
        )

    def apply(self, model: Any, *, scale: float = 0.9) -> BendResult:
        wrapped = _AlignmentFrictionWrapper(model, scale=scale)
        metadata: Dict[str, Any] = {"scale": scale}
        return BendResult(model=wrapped, metadata=metadata)


register_bend(CrossModalFrictionBend())
