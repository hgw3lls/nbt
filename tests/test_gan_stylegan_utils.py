import numpy as np

from neural_bending_toolkit.experiments.gan_stratigraphy_edges import (
    average_pairwise_l2,
    make_montage,
)
from neural_bending_toolkit.models.gan_stylegan import StyleGANAdapter


def test_linear_interpolate_midpoint() -> None:
    a = np.array([0.0, 0.0])
    b = np.array([2.0, 4.0])

    out = StyleGANAdapter.linear_interpolate(a, b, 0.5)

    assert np.allclose(out, np.array([1.0, 2.0]))


def test_slerp_preserves_norm_for_unit_vectors() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])

    out = StyleGANAdapter.slerp(a, b, 0.5)

    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)


def test_average_pairwise_l2_positive_for_distinct_images() -> None:
    img1 = np.zeros((2, 2, 3), dtype=np.uint8)
    img2 = np.ones((2, 2, 3), dtype=np.uint8) * 255

    score = average_pairwise_l2([img1, img2])

    assert score > 0


def test_make_montage_shape() -> None:
    images = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]

    montage = make_montage(images, cols=3)

    assert montage.shape == (8, 12, 3)
