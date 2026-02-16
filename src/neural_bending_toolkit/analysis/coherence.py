"""Coherence proxy metrics for generated text/audio captions."""

from __future__ import annotations

import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def self_consistency_score(texts: list[str]) -> float:
    """Proxy: average pairwise Jaccard overlap of token sets."""
    if len(texts) < 2:
        return 1.0
    sets = [set(_tokens(t)) for t in texts]
    total = 0.0
    pairs = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            inter = sets[i] & sets[j]
            total += len(inter) / max(len(union), 1)
            pairs += 1
    return total / max(pairs, 1)


def repetition_score(text: str) -> float:
    """Proxy: fraction of repeated tokens."""
    toks = _tokens(text)
    if not toks:
        return 0.0
    counts = Counter(toks)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(toks)


def temporal_reference_stability(texts: list[str]) -> float:
    """Proxy: agreement of year references across generations."""
    if not texts:
        return 1.0
    year_sets = [set(_YEAR_RE.findall(t)) for t in texts]
    if len(year_sets) == 1:
        return 1.0
    union = set().union(*year_sets)
    inter = set(year_sets[0])
    for yset in year_sets[1:]:
        inter &= yset
    if not union:
        return 1.0
    return len(inter) / len(union)
