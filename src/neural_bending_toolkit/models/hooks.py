"""Generic module hook helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HookContext:
    """Context passed to hook callbacks."""

    layer_name: str
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)


def register_forward_hook(module: Any, hook_fn: Callable[..., Any]) -> Any:
    """Register a forward hook on a module and return the handle."""

    return module.register_forward_hook(hook_fn)


def register_forward_pre_hook(module: Any, hook_fn: Callable[..., Any]) -> Any:
    """Register a forward pre-hook on a module and return the handle."""

    return module.register_forward_pre_hook(hook_fn)
