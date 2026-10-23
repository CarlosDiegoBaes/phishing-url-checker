"""Held-out metrics on a hand-built score vector, with no dataset."""

from __future__ import annotations

import numpy as np

from phishguard.evaluate import metrics_from_probabilities, split_features


def test_metrics_match_a_hand_counted_confusion_matrix() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.4, 0.9])
    metrics = metrics_from_probabilities(
        labels, probabilities, 0.5, n_train=10, model_name="logistic"
    )
    assert metrics.confusion == [[1, 1], [1, 1]]
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.n_train == 10
    assert metrics.n_test == 4
    assert metrics.threshold == 0.5
    assert metrics.model_name == "logistic"


def test_a_higher_threshold_can_drop_a_borderline_positive() -> None:
    labels = np.array([1, 1])
    probabilities = np.array([0.4, 0.9])
    loose = metrics_from_probabilities(labels, probabilities, 0.3, n_train=1, model_name="logistic")
    strict = metrics_from_probabilities(
        labels, probabilities, 0.5, n_train=1, model_name="logistic"
    )
    assert loose.recall == 1.0
    assert strict.recall == 0.5
    assert strict.confusion == [[0, 0], [1, 1]]


def test_split_is_stratified_and_repeatable() -> None:
    features = np.zeros((100, 2))
    labels = np.array([0] * 80 + [1] * 20)
    x_train, x_test, y_train, y_test = split_features(features, labels)
    assert len(y_test) == 20
    assert int(y_test.sum()) == 4
    assert int(y_train.sum()) == 16
    _, _, _, y_test_again = split_features(features, labels)
    assert np.array_equal(y_test, y_test_again)
