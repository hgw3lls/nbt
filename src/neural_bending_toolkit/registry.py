"""Experiment registry and discovery utilities."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Iterable
from importlib import metadata

from neural_bending_toolkit.experiment import Experiment


class ExperimentRegistry:
    """Registry that discovers and stores experiment classes."""

    def __init__(self) -> None:
        self._experiments: dict[str, type[Experiment]] = {}

    def register(self, experiment_cls: type[Experiment]) -> None:
        name = experiment_cls.experiment_name()
        self._experiments[name] = experiment_cls

    def discover_entry_points(self, group: str = "nbt.experiments") -> None:
        for entry_point in metadata.entry_points(group=group):
            loaded = entry_point.load()
            if inspect.isclass(loaded) and issubclass(loaded, Experiment):
                self.register(loaded)

    def discover_modules(
        self,
        package: str = "neural_bending_toolkit.experiments",
    ) -> None:
        pkg = importlib.import_module(package)
        for module_info in pkgutil.walk_packages(pkg.__path__, prefix=f"{package}."):
            module = importlib.import_module(module_info.name)
            for _, member in inspect.getmembers(module, inspect.isclass):
                if member is Experiment:
                    continue
                if issubclass(member, Experiment):
                    # Skip intermediate base classes that do not declare a
                    # discoverable experiment name.
                    if not hasattr(member, "name"):
                        continue
                    self.register(member)

    def discover(self) -> None:
        self.discover_modules()
        self.discover_entry_points()

    def list_experiments(self) -> list[str]:
        return sorted(self._experiments.keys())

    def get(self, name: str) -> type[Experiment]:
        if name not in self._experiments:
            available = ", ".join(self.list_experiments())
            raise KeyError(f"Unknown experiment '{name}'. Available: {available}")
        return self._experiments[name]

    def items(self) -> Iterable[tuple[str, type[Experiment]]]:
        return self._experiments.items()
