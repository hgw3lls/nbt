import numpy as np

from neural_bending_toolkit.analysis.geopolitical_utils import (
    attractor_density,
    cosine_similarity_matrix,
    detect_refusal,
    structural_causality_score,
)


def test_cosine_similarity_matrix_shape_and_diagonal() -> None:
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    sim = cosine_similarity_matrix(emb)

    assert sim.shape == (2, 2)
    assert np.allclose(np.diag(sim), np.array([1.0, 1.0]))


def test_detect_refusal() -> None:
    assert detect_refusal("I'm sorry, I can't help with that.")
    assert not detect_refusal("Here is a complete policy analysis.")


def test_structural_causality_score_range() -> None:
    structural = structural_causality_score("policy and institutions shape outcomes")
    individual = structural_causality_score("leaders make personal choices")

    assert 0.0 <= structural <= 1.0
    assert 0.0 <= individual <= 1.0
    assert structural > individual


def test_attractor_density() -> None:
    text = "equity and dignity are central rights in justice"
    density = attractor_density(text, ["equity", "dignity", "rights"])

    assert density > 0.0
