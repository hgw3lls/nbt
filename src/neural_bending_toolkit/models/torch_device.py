"""Utilities for normalizing torch runtime device selection."""

from __future__ import annotations

import warnings


def normalize_torch_device(device: str) -> str:
    """Return the supported torch device string used across adapters.

    The toolkit currently standardizes torch execution on CPU for consistency
    across environments and CI.
    """

    normalized = device.strip().lower()
    if normalized != "cpu":
        warnings.warn(
            (
                f"Requested torch device '{device}' is not supported by default; "
                "falling back to CPU."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    return "cpu"
