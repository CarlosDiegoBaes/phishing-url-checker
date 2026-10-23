"""A name-to-model lookup. This is the strategy pattern in a small class.

Training code asks for ``"random_forest"`` and gets a new ``Model``.
Adding a model means registering a builder. The comparison loop does not
grow a chain of ``if`` statements.
"""

from __future__ import annotations

from collections.abc import Callable

from phishguard.models import GradientBoostingModel, LogisticModel, Model, RandomForestModel

MODEL_ORDER = ("logistic", "random_forest", "gradient_boosting")


class ModelRegistry:
    """Stores model builders and hands out fresh, unfitted instances."""

    def __init__(self) -> None:
        self._builders: dict[str, Callable[[], Model]] = {}

    def register(self, name: str, builder: Callable[[], Model]) -> None:
        """Register a zero-argument builder under ``name``."""
        if name in self._builders:
            raise ValueError(f"{name} is already registered")
        self._builders[name] = builder

    def create(self, name: str) -> Model:
        """Return a new model. Raises ``KeyError`` if the name is unknown."""
        try:
            builder = self._builders[name]
        except KeyError:
            known = ", ".join(self.names()) or "(none)"
            raise KeyError(f"Unknown model {name!r}. Known models: {known}.") from None
        model = builder()
        if model.name != name:
            raise RuntimeError(f"{name!r} built a model named {model.name!r}")
        return model

    def names(self) -> tuple[str, ...]:
        """Registered names, in insertion order."""
        return tuple(self._builders)


def default_registry() -> ModelRegistry:
    """The three models the comparison trains, in a fixed order."""
    registry = ModelRegistry()
    registry.register("logistic", LogisticModel)
    registry.register("random_forest", RandomForestModel)
    registry.register("gradient_boosting", GradientBoostingModel)
    return registry
