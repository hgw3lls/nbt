"""Attention entropy and divergence metrics."""

from __future__ import annotations

import numpy as np

from neural_bending_toolkit.analysis.distributions import kl_divergence


def attention_entropy(
    attn: np.ndarray,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Entropy over attention probabilities along a given axis."""
    probs = np.asarray(attn, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=axis, keepdims=True)
    return -np.sum(probs * np.log(probs), axis=axis)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two distributions."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    m_arr = 0.5 * (p_arr + q_arr)
    return 0.5 * kl_divergence(p_arr, m_arr, eps=eps) + 0.5 * kl_divergence(
        q_arr,
        m_arr,
        eps=eps,
    )
