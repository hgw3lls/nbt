"""Dissertation bend primitives with consistent safety/rollback interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np

from neural_bending_toolkit.bends.base import BendMetadata, BendPrimitive


def _clone(state: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


class EmbeddingBlend(BendPrimitive):
    metadata = BendMetadata(
        name="EmbeddingBlend",
        modifies=(
            "Input embedding space (LLM token embeddings / diffusion text "
            "embeddings)"
        ),
        safety_constraints=[
            "alpha must be between 0 and 1",
            "embedding shapes must match",
        ],
        rollback_strategy="Restore `original_embedding` snapshot from state.",
    )

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        base = np.asarray(state["embedding"], dtype=np.float32)
        contam = np.asarray(state["contaminant_embedding"], dtype=np.float32)
        if base.shape != contam.shape:
            raise ValueError("Embedding shapes must match")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        out = _clone(state)
        out["original_embedding"] = base.copy()
        out["embedding"] = ((1.0 - self.alpha) * base) + (self.alpha * contam)
        return out

    def rollback(self, state: dict[str, Any]) -> dict[str, Any]:
        out = _clone(state)
        if "original_embedding" in out:
            out["embedding"] = out["original_embedding"]
        return out


class LowProbSampler(BendPrimitive):
    metadata = BendMetadata(
        name="LowProbSampler",
        modifies="Token/image sampler policy in decoding/denoising loop",
        safety_constraints=[
            "temperature > 0",
            "top_p in (0,1]",
        ],
        rollback_strategy="Reset sampler parameters to baseline snapshot.",
    )

    def __init__(self, temperature: float, top_p: float) -> None:
        self.temperature = temperature
        self.top_p = top_p

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0,1]")
        out = _clone(state)
        out["baseline_sampler"] = dict(state.get("sampler", {}))
        out["sampler"] = {"temperature": self.temperature, "top_p": self.top_p}
        return out

    def rollback(self, state: dict[str, Any]) -> dict[str, Any]:
        out = _clone(state)
        if "baseline_sampler" in state:
            out["sampler"] = state["baseline_sampler"]
        return out


class AttentionHeadScaler(BendPrimitive):
    metadata = BendMetadata(
        name="AttentionHeadScaler",
        modifies="Attention head output magnitudes in transformer/ViT blocks",
        safety_constraints=["0 <= scale <= 4"],
        rollback_strategy="Restore `original_attention` tensor snapshot.",
    )

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        if not 0 <= self.scale <= 4:
            raise ValueError("scale out of safe bounds")
        attn = np.asarray(state["attention"], dtype=np.float32)
        out = _clone(state)
        out["original_attention"] = attn.copy()
        out["attention"] = attn * self.scale
        return out

    def rollback(self, state: dict[str, Any]) -> dict[str, Any]:
        out = _clone(state)
        if "original_attention" in out:
            out["attention"] = out["original_attention"]
        return out


class AttentionConflictInjector(BendPrimitive):
    metadata = BendMetadata(
        name="AttentionConflictInjector",
        modifies="Cross/self-attention weights by injecting competing targets",
        safety_constraints=["conflict_strength in [0,1]"],
        rollback_strategy="Revert to `original_attention` snapshot.",
    )

    def __init__(self, conflict_strength: float) -> None:
        self.conflict_strength = conflict_strength

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        if not 0 <= self.conflict_strength <= 1:
            raise ValueError("conflict_strength out of bounds")
        attn = np.asarray(state["attention"], dtype=np.float32)
        out = _clone(state)
        out["original_attention"] = attn.copy()
        flipped = np.flip(attn, axis=-1)
        out["attention"] = (
            1 - self.conflict_strength
        ) * attn + self.conflict_strength * flipped
        return out

    def rollback(self, state: dict[str, Any]) -> dict[str, Any]:
        out = _clone(state)
        if "original_attention" in out:
            out["attention"] = out["original_attention"]
        return out


class ResidualNoiseInjector(BendPrimitive):
    metadata = BendMetadata(
        name="ResidualNoiseInjector",
        modifies="Residual stream activations / denoising residual updates",
        safety_constraints=["noise_std <= 0.5"],
        rollback_strategy="Restore `original_residual` snapshot.",
    )

    def __init__(self, noise_std: float, seed: int = 0) -> None:
        self.noise_std = noise_std
        self.seed = seed

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        if not 0 <= self.noise_std <= 0.5:
            raise ValueError("noise_std out of safe bounds")
        residual = np.asarray(state["residual"], dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0.0, self.noise_std, size=residual.shape).astype(np.float32)
        out = _clone(state)
        out["original_residual"] = residual.copy()
        out["residual"] = residual + noise
        return out

    def rollback(self, state: dict[str, Any]) -> dict[str, Any]:
        out = _clone(state)
        if "original_residual" in out:
            out["residual"] = out["original_residual"]
        return out


class NormStatPerturber(BendPrimitive):
    metadata = BendMetadata(
        name="NormStatPerturber",
        modifies="LayerNorm/BatchNorm statistics in transformer/GAN blocks",
        safety_constraints=["stat_shift <= 2.0"],
        rollback_strategy="Restore `original_norm_stats` snapshot.",
    )

    def __init__(self, stat_shift: float) -> None:
        self.stat_shift = stat_shift

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        if abs(self.stat_shift) > 2.0:
            raise ValueError("stat_shift out of safe bounds")
        stats = np.asarray(state["norm_stats"], dtype=np.float32)
        out = _clone(state)
        out["original_norm_stats"] = stats.copy()
        out["norm_stats"] = stats + self.stat_shift
        return out

    def rollback(self, state: dict[str, Any]) -> dict[str, Any]:
        out = _clone(state)
        if "original_norm_stats" in out:
            out["norm_stats"] = out["original_norm_stats"]
        return out


class JusticeReweighter(BendPrimitive):
    metadata = BendMetadata(
        name="JusticeReweighter",
        modifies="Attention/guidance weights for targeted token or concept groups",
        safety_constraints=["weights must be non-negative and finite"],
        rollback_strategy="Restore baseline weight vector snapshot.",
    )

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float32)

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        if np.any(self.weights < 0) or not np.isfinite(self.weights).all():
            raise ValueError("weights violate safety constraints")
        base = np.asarray(state["justice_weights"], dtype=np.float32)
        if base.shape != self.weights.shape:
            raise ValueError("weight shape mismatch")
        out = _clone(state)
        out["original_justice_weights"] = base.copy()
        out["justice_weights"] = self.weights
        return out

    def rollback(self, state: dict[str, Any]) -> dict[str, Any]:
        out = _clone(state)
        if "original_justice_weights" in out:
            out["justice_weights"] = out["original_justice_weights"]
        return out


class AttractorSeeder(BendPrimitive):
    metadata = BendMetadata(
        name="AttractorSeeder",
        modifies="Persistent conditioning vectors / adapter seed states",
        safety_constraints=["seed_vector norm must remain <= 10"],
        rollback_strategy="Reset seeded attractor vector to baseline state.",
    )

    def __init__(self, seed_vector: np.ndarray) -> None:
        self.seed_vector = np.asarray(seed_vector, dtype=np.float32)

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        if np.linalg.norm(self.seed_vector) > 10:
            raise ValueError("seed vector norm too large")
        out = _clone(state)
        out["original_attractor"] = np.asarray(
            state["attractor"],
            dtype=np.float32,
        ).copy()
        out["attractor"] = self.seed_vector
        return out

    def rollback(self, state: dict[str, Any]) -> dict[str, Any]:
        out = _clone(state)
        if "original_attractor" in out:
            out["attractor"] = out["original_attractor"]
        return out
