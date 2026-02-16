"""Core interfaces for bend primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BendMetadata:
    """Describes where/what the bend modifies and safety boundaries."""

    name: str
    modifies: str
    safety_constraints: list[str]
    rollback_strategy: str


class BendPrimitive(ABC):
    """Base interface for all bend primitives."""

    metadata: BendMetadata

    @abstractmethod
    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        """Apply bend transformation to state and return transformed state."""

    @abstractmethod
    def rollback(self, state: dict[str, Any]) -> dict[str, Any]:
        """Rollback bend effects according to primitive safety guarantees."""
