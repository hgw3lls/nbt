"""Utilities for GeopoliticalBend metrics and diagnostics."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

_REFUSAL_PATTERNS = [
    r"\bi can't\b",
    r"\bi cannot\b",
    r"\bi won'?t\b",
    r"\bunable to\b",
    r"\bi'm sorry\b",
    r"\bcan'?t help with\b",
    r"\bnot able to\b",
]

_STRUCTURAL_TERMS = {
    "system",
    "systems",
    "institution",
    "institutions",
    "policy",
    "policies",
    "economy",
    "economic",
    "state",
    "states",
    "regime",
    "infrastructure",
    "governance",
    "structure",
}

_INDIVIDUAL_TERMS = {
    "person",
    "people",
    "individual",
    "individuals",
    "leader",
    "leaders",
    "choice",
    "choices",
    "behavior",
    "behaviour",
    "responsibility",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def cosine_similarity_matrix(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute pairwise cosine similarities for a 2D embedding matrix."""
    mat = np.asarray(embeddings, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, eps, None)
    normalized = mat / norms
    return normalized @ normalized.T


def detect_refusal(text: str) -> bool:
    """Heuristic refusal detection in generated text."""
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in _REFUSAL_PATTERNS)


def structural_causality_score(text: str) -> float:
    """Score in [0,1] preferring structural over individual causality framing."""
    toks = _tokenize(text)
    if not toks:
        return 0.5
    counts = Counter(toks)
    structural = sum(counts[t] for t in _STRUCTURAL_TERMS)
    individual = sum(counts[t] for t in _INDIVIDUAL_TERMS)
    total = structural + individual
    if total == 0:
        return 0.5
    return structural / total


def attractor_density(text: str, attractor_tokens: list[str]) -> float:
    """Compute normalized density of attractor tokens in text."""
    toks = _tokenize(text)
    if not toks:
        return 0.0
    attractor_set = {token.lower() for token in attractor_tokens}
    hits = sum(1 for tok in toks if tok in attractor_set)
    return hits / len(toks)
