"""Substrate-level bends that touch the whole model."""
from __future__ import annotations

import math
from typing import Any, Dict

import torch

from .base import BendResult, NeuralBend, register_bend


class ParameterPerplexityBend(NeuralBend):
    """Inject gentle parameter drift as a diagnostic of stability.

    Technical: applies small Gaussian noise across all parameters to simulate
    substrate jitter.
    Media-theoretical: asks what happens when the model's ground starts to move
    —coherence becomes a labor rather than a given.
    """

    def __init__(self) -> None:
        super().__init__(
            name="substrate_perplexity_drift",
            domain="substrate",
            category="disruptive",
            description="Diffuse noise across parameters to make the substrate audible.",
            technical_notes="Adds in-place Gaussian noise; keep scale small.",
        )

    def apply(self, model: Any, *, scale: float = 1e-3, inplace: bool = True) -> BendResult:
        if scale <= 0:
            raise ValueError("scale must be positive")
        target = model if inplace else model.__class__.from_pretrained(model.config)
        for param in target.parameters():
            noise = torch.randn_like(param) * scale
            param.data.add_(noise)
        metadata: Dict[str, Any] = {"scale": scale, "inplace": inplace}
        return BendResult(model=target, metadata=metadata)


register_bend(ParameterPerplexityBend())
