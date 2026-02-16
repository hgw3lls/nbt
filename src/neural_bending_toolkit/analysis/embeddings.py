"""Embedding topology utilities (PCA/UMAP)."""

from __future__ import annotations

import numpy as np


def compute_pca_projection(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project embeddings to 2D/3D with PCA via SVD."""
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be 2D")
    x_centered = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    return u[:, :n_components] * s[:n_components]


def compute_umap_projection(
    embeddings: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """UMAP projection if available; fallback to PCA otherwise."""
    x = np.asarray(embeddings, dtype=np.float64)
    try:
        import umap

        reducer = umap.UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(x)
    except Exception:
        return compute_pca_projection(x, n_components=n_components)
