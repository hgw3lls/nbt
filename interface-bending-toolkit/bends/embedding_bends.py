"""Embedding-space bends that contaminate or invert concepts."""
from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch import nn

from .base import BendResult, NeuralBend, register_bend


def _token_ids(tokenizer: Any, text: str) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Token '{text}' produced no ids for tokenizer {getattr(tokenizer, 'name_or_path', 'unknown')}")
    return ids


def _pooled_embedding(embedding_matrix: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    rows = embedding_matrix[torch.tensor(ids, device=embedding_matrix.device)]
    return rows.mean(dim=0)


class EmbeddingDriftBend(NeuralBend):
    """Cross-wires meanings by drifting or reflecting selected embeddings.

    Technical: re-centers source token embeddings toward target tokens with an
    adjustable mix; mirrors past drift experiments so glitch remains audible.
    Media-theoretical: surfaces how "meaning" is a negotiated geometry in
    embedding space—here we deliberately leak one semantic cluster into another
    to reveal the politics of proximity.
    """

    def __init__(self) -> None:
        super().__init__(
            name="embedding_contamination",
            domain="embedding",
            category="revelatory",
            description="Blend or reflect concept vectors to expose how closeness encodes normativity.",
            technical_notes="Requires a tokenizer and access to model.get_input_embeddings().weight.",
        )

    def apply(
        self,
        model: Any,
        *,
        tokenizer: Any,
        source_tokens: Iterable[str],
        target_tokens: Iterable[str],
        mode: str = "drift",
        alpha: float = 0.5,
        mix: float = 1.0,
        inplace: bool = False,
    ) -> BendResult:
        if mode not in {"drift", "invert"}:
            raise ValueError("mode must be 'drift' or 'invert'")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 0.0 <= mix <= 1.0:
            raise ValueError("mix must be between 0 and 1")

        src_list = list(source_tokens)
        tgt_list = list(target_tokens)
        if len(src_list) != len(tgt_list):
            raise ValueError("source_tokens and target_tokens must be the same length")

        base_model = model if inplace else copy.deepcopy(model)
        embedding_matrix: torch.Tensor = base_model.get_input_embeddings().weight
        modified = embedding_matrix.detach().clone()

        metrics: List[Dict[str, float]] = []
        for src, tgt in zip(src_list, tgt_list):
            src_ids = _token_ids(tokenizer, src)
            tgt_ids = _token_ids(tokenizer, tgt)
            src_vec = _pooled_embedding(modified, src_ids)
            tgt_vec = _pooled_embedding(modified, tgt_ids)

            if mode == "drift":
                delta = alpha * (tgt_vec - src_vec)
            else:
                center = 0.5 * (src_vec + tgt_vec)
                reflected = 2 * center - src_vec
                delta = alpha * (reflected - src_vec)

            bent_vec = src_vec + delta
            for idx in src_ids:
                modified[idx] = (1 - mix) * embedding_matrix[idx] + mix * bent_vec

            before_cos = nn.functional.cosine_similarity(src_vec, tgt_vec, dim=0).item()
            after_cos = nn.functional.cosine_similarity(bent_vec, tgt_vec, dim=0).item()
            metrics.append(
                {
                    "source": src,
                    "target": tgt,
                    "before_cos": before_cos,
                    "after_cos": after_cos,
                    "drift_magnitude": torch.norm(delta).item(),
                }
            )

        embedding_matrix.data.copy_(modified)
        return BendResult(model=base_model, metadata={"metrics": metrics, "mode": mode})


register_bend(EmbeddingDriftBend())
