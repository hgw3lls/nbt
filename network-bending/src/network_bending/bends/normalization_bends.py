"""Layer normalization bends that redraw what counts as "normal.""" 
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn

from .base import BendResult, NeuralBend, register_bend


def _parse_token_list(raw: str) -> List[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def _care_centroid(tokenizer: Any, tokens: Iterable[str], embedding_matrix: torch.Tensor) -> torch.Tensor:
    ids: List[int] = []
    for tok in tokens:
        encoded = tokenizer.encode(tok, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Token '{tok}' produced no ids for tokenizer {getattr(tokenizer, 'name_or_path', 'unknown')}")
        ids.extend(encoded)
    rows = embedding_matrix[torch.tensor(ids, device=embedding_matrix.device)]
    centroid = rows.mean(dim=0)
    norm = torch.norm(centroid) + 1e-8
    return centroid / norm


def _similarity_vectors(embedding_matrix: torch.Tensor, centroid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    normed_embed = embedding_matrix / (embedding_matrix.norm(dim=1, keepdim=True) + 1e-8)
    dot_sim = torch.matmul(normed_embed, centroid)
    distance = torch.norm(normed_embed - centroid, dim=1)
    return dot_sim, distance


def _bend_logits(
    logits: torch.Tensor,
    mode: str,
    care_scale: float,
    variance_scale: float,
    similarity: torch.Tensor,
    distance: torch.Tensor,
    mix: float,
) -> torch.Tensor:
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")
    bent = logits
    if mode == "care_center":
        bent = logits + care_scale * similarity
    elif mode == "variance":
        noise = torch.randn_like(logits) * variance_scale * (1 + distance)
        bent = logits + noise
    return (1 - mix) * logits + mix * bent


class _LayerNormWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        mode: str,
        care_scale: float,
        variance_scale: float,
        mix: float,
        similarity: torch.Tensor,
        distance: torch.Tensor,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.mode = mode
        self.care_scale = care_scale
        self.variance_scale = variance_scale
        self.mix = mix
        self.register_buffer("similarity", similarity)
        self.register_buffer("distance", distance)

    def forward(self, input_ids=None, **kwargs):  # type: ignore[override]
        outputs = self.base_model(input_ids=input_ids, **kwargs)
        logits = outputs.logits
        adjusted = _bend_logits(
            logits=logits,
            mode=self.mode,
            care_scale=self.care_scale,
            variance_scale=self.variance_scale,
            similarity=self.similarity,
            distance=self.distance,
            mix=self.mix,
        )
        return outputs.__class__(logits=adjusted, **outputs.to_tuple()[1:]) if hasattr(outputs, "to_tuple") else outputs


class LayerNormCareBend(NeuralBend):
    """Tweak layer norm reference to center care and variance.

    Technical: biases logits based on similarity to a care vocabulary or jitters
    variance to show normalization as an active choice.
    Media-theoretical: frames normalization as governance—what counts as
    "average" is bent toward repair rather than control.
    """

    def __init__(self) -> None:
        super().__init__(
            name="layernorm_care_center",
            domain="normalization",
            category="recoherent",
            description="Perturb norm reference so care-centered words become the statistical anchor.",
            technical_notes="Wraps logits with similarity-weighted bias or variance noise.",
        )

    def apply(
        self,
        model: Any,
        *,
        tokenizer: Any,
        mode: str = "care_center",
        care_tokens: str = "care,repair,collective,community",
        care_scale: float = 5.0,
        variance_scale: float = 1.0,
        mix: float = 1.0,
    ) -> BendResult:
        if mode not in {"care_center", "variance"}:
            raise ValueError("mode must be 'care_center' or 'variance'")
        care_list = _parse_token_list(care_tokens)
        embedding_matrix = model.get_input_embeddings().weight.detach()
        centroid = _care_centroid(tokenizer, care_list, embedding_matrix)
        similarity, distance = _similarity_vectors(embedding_matrix, centroid)

        wrapped = _LayerNormWrapper(
            base_model=model,
            mode=mode,
            care_scale=care_scale,
            variance_scale=variance_scale,
            mix=mix,
            similarity=similarity,
            distance=distance,
        )
        metadata: Dict[str, Any] = {"mode": mode, "care_tokens": care_list, "care_scale": care_scale, "variance_scale": variance_scale}
        return BendResult(model=wrapped, metadata=metadata)


register_bend(LayerNormCareBend())
