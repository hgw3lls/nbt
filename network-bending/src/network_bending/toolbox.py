"""User-facing API for the Network/Neural Bending toolbox.

This layer keeps discovery simple while foregrounding the philosophical
categories that animate each bend.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .bends import BendResult, get_bend, list_bends


def list_bend_summaries(domain: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, str]]:
    """Return human-readable summaries of registered bends."""

    return [bend.summary() for bend in list_bends(domain=domain, category=category)]


def apply_bend(bend_name: str, model: Any, *, return_metadata: bool = False, **kwargs: Any):
    """Apply a named bend to ``model`` and optionally return metadata.

    Parameters
    ----------
    bend_name: str
        Registered bend identifier.
    model: Any
        The model to bend.
    return_metadata: bool
        When True, return a :class:`BendResult` instead of just the bent model.
    **kwargs: Any
        Passed through to the bend's ``apply`` method.
    """

    bend = get_bend(bend_name)
    result: BendResult = bend(model, **kwargs)
    return result if return_metadata else result.model


__all__ = ["list_bend_summaries", "apply_bend"]
