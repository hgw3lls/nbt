"""Distribution-level metrics."""

from __future__ import annotations

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute KL(P || Q) for categorical distributions."""
    p_safe = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    q_safe = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0)
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))
