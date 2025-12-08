"""Residual-stream bends that bottleneck or nudge narrative flow."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import torch
from torch import nn

from .base import BendResult, NeuralBend, register_bend


def _bottleneck(hidden_states: torch.Tensor, bottleneck_dim: int, alpha: float) -> torch.Tensor:
    hidden_dim = hidden_states.size(-1)
    dim = max(1, min(bottleneck_dim, hidden_dim))
    projector = torch.eye(hidden_dim, device=hidden_states.device)[:dim]
    compressed = torch.matmul(hidden_states, projector.t())
    restored = torch.matmul(compressed, projector)
    return hidden_states + alpha * (restored - hidden_states)


def _relational_bend(hidden_states: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    return hidden_states + alpha * direction


class _ResidualWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        mode: str,
        bottleneck_dim: int,
        alpha: float,
        mix: float,
        direction: torch.Tensor,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.mode = mode
        self.bottleneck_dim = bottleneck_dim
        self.alpha = alpha
        self.mix = mix
        self.register_buffer("direction", direction)

    def forward(self, input_ids=None, **kwargs):  # type: ignore[override]
        outputs = self.base_model(input_ids=input_ids, output_hidden_states=True, **kwargs)
        hidden = outputs.hidden_states[-1]

        if self.mode == "bottleneck":
            bent = _bottleneck(hidden, self.bottleneck_dim, self.alpha)
        else:
            bent = _relational_bend(hidden, self.direction, self.alpha)

        mixed_hidden = (1 - self.mix) * hidden + self.mix * bent
        logits = self.base_model.lm_head(mixed_hidden)

        # Return a lightweight object with logits for compatibility
        return SimpleNamespace(logits=logits, hidden_states=outputs.hidden_states)


class ResidualBottleneckBend(NeuralBend):
    """Bottleneck the residual stream to expose collapse into minimal forms.

    Technical: compresses the residual channel space before projecting to logits,
    forcing the model to breathe through a narrower grammar.
    Media-theoretical: stages failure as a probe—showing how coherence is a thin
    membrane stretched across the residual spine.
    """

    def __init__(self) -> None:
        super().__init__(
            name="residual_bottleneck_minimal_forms",
            domain="residual",
            category="disruptive",
            description="Throttle residual bandwidth to watch meaning collapse and reassemble.",
            technical_notes="Wraps causal LM forward pass; requires lm_head and hidden states.",
        )

    def apply(
        self,
        model: Any,
        *,
        mode: str = "bottleneck",
        bottleneck_dim: int = 64,
        alpha: float = 0.5,
        mix: float = 1.0,
        direction: torch.Tensor | None = None,
    ) -> BendResult:
        if mode not in {"bottleneck", "relational"}:
            raise ValueError("mode must be 'bottleneck' or 'relational'")
        if not hasattr(model, "lm_head"):
            raise AttributeError("Model must expose an lm_head for residual bending")
        if direction is None:
            direction = torch.zeros(1, 1, model.config.hidden_size, device=next(model.parameters()).device)

        wrapped = _ResidualWrapper(model, mode, bottleneck_dim, alpha, mix, direction)
        metadata: Dict[str, Any] = {"mode": mode, "bottleneck_dim": bottleneck_dim, "alpha": alpha, "mix": mix}
        return BendResult(model=wrapped, metadata=metadata)


register_bend(ResidualBottleneckBend())
