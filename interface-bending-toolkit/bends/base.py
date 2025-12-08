"""Foundations for media-theoretical neural bending.

Defines the :class:`NeuralBend` base class and a small registry so that
bends can self-describe across domains and meta-categories.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

VALID_CATEGORIES = {"revelatory", "disruptive", "recoherent"}


@dataclass
class BendResult:
    """Container for a bent model plus optional metadata."""

    model: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class NeuralBend(ABC):
    """Abstract base for philosophically-aware bends.

    Each bend declares the architectural domain it touches and a
    meta-category describing whether it reveals, disrupts, or recoheres
    the model's internal equilibrium.
    """

    name: str
    domain: str
    category: str
    description: str
    technical_notes: str

    def __init__(
        self,
        name: str,
        domain: str,
        category: str,
        description: str,
        technical_notes: str,
    ) -> None:
        if category not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}")
        self.name = name
        self.domain = domain
        self.category = category
        self.description = description
        self.technical_notes = technical_notes

    def __call__(self, model: Any, **kwargs: Any) -> BendResult:
        return self.apply(model, **kwargs)

    @abstractmethod
    def apply(self, model: Any, **kwargs: Any) -> BendResult:
        """Run the bend on ``model`` and return a :class:`BendResult`."""
        raise NotImplementedError

    def summary(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "domain": self.domain,
            "category": self.category,
            "description": self.description,
        }


_BEND_REGISTRY: Dict[str, NeuralBend] = {}


def register_bend(bend: NeuralBend) -> NeuralBend:
    """Register a bend instance for discovery."""

    if bend.name in _BEND_REGISTRY:
        raise ValueError(f"Bend '{bend.name}' already registered")
    _BEND_REGISTRY[bend.name] = bend
    return bend


def get_bend(name: str) -> NeuralBend:
    if name not in _BEND_REGISTRY:
        raise KeyError(f"No bend named '{name}'. Known bends: {list(_BEND_REGISTRY)}")
    return _BEND_REGISTRY[name]


def list_bends(
    domain: Optional[str] = None, category: Optional[str] = None
) -> List[NeuralBend]:
    bends: Iterable[NeuralBend] = _BEND_REGISTRY.values()
    if domain is not None:
        bends = [b for b in bends if b.domain == domain]
    if category is not None:
        bends = [b for b in bends if b.category == category]
    return list(bends)
