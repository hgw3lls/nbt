"""Positional encoding bends that warp machine temporality."""
from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torch import nn

from .base import BendResult, NeuralBend, register_bend


class _PositionalWarpWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, warp_strength: float, scramble: bool) -> None:
        super().__init__()
        self.base_model = base_model
        self.warp_strength = warp_strength
        self.scramble = scramble

    def forward(self, input_ids=None, position_ids=None, **kwargs):  # type: ignore[override]
        if position_ids is None and input_ids is not None:
            seq_len = input_ids.shape[1]
            base_positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        else:
            base_positions = position_ids

        if base_positions is None:
            return self.base_model(input_ids=input_ids, position_ids=position_ids, **kwargs)

        if self.scramble:
            noise = torch.randint(0, seq_len := base_positions.shape[1], base_positions.shape, device=base_positions.device)
            warped_positions = (base_positions + noise) % max(seq_len, 1)
        else:
            time = torch.linspace(0, math.pi, steps=base_positions.shape[1], device=base_positions.device)
            warp_curve = torch.sin(time) * self.warp_strength
            warped_positions = base_positions + warp_curve

        return self.base_model(input_ids=input_ids, position_ids=warped_positions.long(), **kwargs)


class TemporalScrambleBend(NeuralBend):
    """Warp positional encoding to reveal the model's sense of time.

    Technical: perturbs position indices with sinusoidal drift or random
    scrambles so attention has to negotiate distorted temporality.
    Media-theoretical: treats position as a politics of ordering—who speaks
    first, who follows, and how chronology fractures under pressure.
    """

    def __init__(self) -> None:
        super().__init__(
            name="positional_temporal_scramble",
            domain="positional",
            category="disruptive",
            description="Warp position ids to expose the model's constructed sense of sequence.",
            technical_notes="Wraps model forward and mutates position_ids before call.",
        )

    def apply(
        self,
        model: Any,
        *,
        warp_strength: float = 3.0,
        scramble: bool = False,
    ) -> BendResult:
        wrapped = _PositionalWarpWrapper(model, warp_strength=warp_strength, scramble=scramble)
        metadata: Dict[str, Any] = {"warp_strength": warp_strength, "scramble": scramble}
        return BendResult(model=wrapped, metadata=metadata)


register_bend(TemporalScrambleBend())
