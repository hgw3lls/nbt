import numpy as np

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


def _state() -> dict:
    return {
        "embedding": np.ones((4,), dtype=np.float32),
        "contaminant_embedding": np.zeros((4,), dtype=np.float32),
        "sampler": {"temperature": 1.0, "top_p": 1.0},
        "attention": np.ones((2, 2), dtype=np.float32),
        "residual": np.ones((4,), dtype=np.float32),
        "norm_stats": np.ones((4,), dtype=np.float32),
        "justice_weights": np.ones((8,), dtype=np.float32),
        "attractor": np.zeros((8,), dtype=np.float32),
    }


def test_primitives_apply_and_rollback() -> None:
    base = _state()
    primitives = [
        EmbeddingBlend(alpha=0.5),
        LowProbSampler(temperature=0.7, top_p=0.9),
        AttentionHeadScaler(scale=1.2),
        AttentionConflictInjector(conflict_strength=0.4),
        ResidualNoiseInjector(noise_std=0.1, seed=1),
        NormStatPerturber(stat_shift=0.2),
        JusticeReweighter(weights=np.ones((8,), dtype=np.float32) * 1.1),
        AttractorSeeder(seed_vector=np.ones((8,), dtype=np.float32) * 0.2),
    ]

    for primitive in primitives:
        bent = primitive.apply(base)
        rolled = primitive.rollback(bent)
        assert isinstance(rolled, dict)
        assert primitive.metadata.modifies
        assert primitive.metadata.safety_constraints
        assert primitive.metadata.rollback_strategy
