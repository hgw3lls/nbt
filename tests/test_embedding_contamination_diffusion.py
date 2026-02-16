import numpy as np

from neural_bending_toolkit.experiments.embedding_contamination_diffusion import (
    blend_embeddings,
)


def test_blend_embeddings_weighted_average() -> None:
    base = np.array([[[1.0, 2.0]]])
    contaminant = np.array([[[3.0, 6.0]]])

    mixed = blend_embeddings(base, contaminant, alpha=0.25)

    assert np.allclose(mixed, np.array([[[1.5, 3.0]]]))
