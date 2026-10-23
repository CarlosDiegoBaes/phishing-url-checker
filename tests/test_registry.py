"""Model registry and the three scikit-learn strategies, on synthetic URLs."""

from __future__ import annotations

import pytest

from phishguard.config import DEFAULT_MODEL, DEFAULT_THRESHOLD
from phishguard.models import GradientBoostingModel, RandomForestModel
from phishguard.pipeline import Pipeline
from phishguard.registry import MODEL_ORDER, ModelRegistry, default_registry


def test_frozen_default_is_a_registered_model() -> None:
    assert DEFAULT_MODEL in default_registry().names()
    assert 0.0 <= DEFAULT_THRESHOLD <= 1.0


def test_default_registry_lists_the_three_models() -> None:
    registry = default_registry()
    assert registry.names() == MODEL_ORDER


def test_create_returns_a_fresh_unfitted_model() -> None:
    registry = default_registry()
    first = registry.create("random_forest")
    second = registry.create("random_forest")
    assert first is not second
    assert first.name == "random_forest"
    with pytest.raises(RuntimeError):
        first.predict_proba([[0.0] * 10])


def test_unknown_model_names_the_ones_that_exist() -> None:
    with pytest.raises(KeyError, match="gradient_boosting"):
        default_registry().create("neural_net")


def test_registering_twice_is_rejected() -> None:
    registry = ModelRegistry()
    registry.register("logistic", lambda: default_registry().create("logistic"))
    with pytest.raises(ValueError):
        registry.register("logistic", lambda: default_registry().create("logistic"))


@pytest.mark.parametrize("name", MODEL_ORDER)
def test_each_model_scores_between_zero_and_one(
    name: str,
    training_urls: list[str],
    training_labels: list[int],
) -> None:
    pipeline = Pipeline(model=default_registry().create(name))
    pipeline.fit(training_urls, training_labels)
    phish = pipeline.score("http://paypal-account-verify.secure-login.tk/signin")
    safe = pipeline.score("https://www.example.com")
    assert 0.0 <= phish.probability <= 1.0
    assert 0.0 <= safe.probability <= 1.0
    assert phish.probability > safe.probability


def test_tree_models_have_no_single_weight() -> None:
    forest = RandomForestModel()
    boosting = GradientBoostingModel()
    assert forest.coefficients() is None
    assert boosting.coefficients() is None
    assert forest.explain([[0.0]]) is None
    assert boosting.explain([[0.0]]) is None
