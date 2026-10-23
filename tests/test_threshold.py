"""Threshold selection on hand-counted scores. No dataset and no fitting."""

from __future__ import annotations

import numpy as np

from phishguard.compare import select_winner
from phishguard.evaluate import choose_threshold, metrics_from_probabilities
from phishguard.verdict import Metrics


def _metrics(name: str, precision: float, recall: float, threshold: float = 0.5) -> Metrics:
    return Metrics(
        accuracy=0.0,
        precision=precision,
        recall=recall,
        f1=0.0,
        confusion=[[0, 0], [0, 0]],
        n_train=1,
        n_test=1,
        threshold=threshold,
        model_name=name,
    )


def test_choose_threshold_raises_recall_and_keeps_the_precision_floor() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.10, 0.20, 0.30, 0.55, 0.40, 0.60, 0.70, 0.90])
    chosen = choose_threshold(labels, probabilities, n_train=8, model_name="logistic")
    assert chosen.threshold == 0.40
    assert chosen.precision == 0.8
    assert chosen.recall == 1.0
    checked = metrics_from_probabilities(labels, probabilities, 0.40, 8, "logistic")
    assert chosen == checked


def test_choose_threshold_falls_back_to_f1_when_precision_is_unreachable() -> None:
    labels = np.array([0, 1, 1])
    probabilities = np.array([0.95, 0.90, 0.10])
    chosen = choose_threshold(
        labels,
        probabilities,
        n_train=3,
        model_name="logistic",
        min_precision=0.99,
    )
    assert chosen.threshold == 0.10
    assert chosen.recall == 1.0
    assert chosen.precision == 2 / 3


def test_winner_is_the_highest_validation_recall_above_the_floor() -> None:
    validation = {
        "logistic": _metrics("logistic", precision=0.90, recall=0.60),
        "random_forest": _metrics("random_forest", precision=0.81, recall=0.75),
        "gradient_boosting": _metrics("gradient_boosting", precision=0.70, recall=0.95),
    }
    assert select_winner(validation) == "random_forest"


def test_winner_tie_prefers_the_simpler_model() -> None:
    validation = {
        "logistic": _metrics("logistic", precision=0.85, recall=0.80),
        "random_forest": _metrics("random_forest", precision=0.85, recall=0.80),
        "gradient_boosting": _metrics("gradient_boosting", precision=0.85, recall=0.80),
    }
    assert select_winner(validation) == "logistic"
