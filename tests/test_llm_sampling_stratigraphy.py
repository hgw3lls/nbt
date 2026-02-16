import numpy as np

from neural_bending_toolkit.experiments.llm_sampling_stratigraphy import kl_divergence


def test_kl_divergence_is_zero_for_identical_distributions() -> None:
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.2, 0.3, 0.5])

    result = kl_divergence(p, q)

    assert abs(result) < 1e-10
