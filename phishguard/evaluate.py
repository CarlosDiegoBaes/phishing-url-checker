"""Split data and turn probabilities into accuracy, precision, recall, and F1.

These functions only see arrays. They do not know which model produced the
probabilities, so the same held-out split can score every model the same way.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from phishguard.verdict import Metrics

SEED = 42
TEST_SIZE = 0.2
MIN_PRECISION = 0.80
THRESHOLDS = tuple(step / 100 for step in range(5, 96))


def split_features(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split.

    Seed 42 and a 20% test set are the split behind the published baseline.
    The same feature matrix and labels produce the same rows every time.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return x_train, x_test, y_train, y_test


def metrics_from_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    n_train: int,
    model_name: str,
) -> Metrics:
    """Score a holdout set at one decision threshold.

    A URL is predicted phishing when its probability is at least ``threshold``.
    Precision, recall, and F1 are for the phishing class.
    """
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    return Metrics(
        accuracy=float(accuracy_score(labels, predicted)),
        precision=float(precision_score(labels, predicted, zero_division=0)),
        recall=float(recall_score(labels, predicted, zero_division=0)),
        f1=float(f1_score(labels, predicted, zero_division=0)),
        confusion=matrix.astype(int).tolist(),
        n_train=n_train,
        n_test=int(len(labels)),
        threshold=threshold,
        model_name=model_name,
    )


def choose_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_train: int,
    model_name: str,
    min_precision: float = MIN_PRECISION,
) -> Metrics:
    """Pick a threshold that raises recall while precision stays reasonable.

    ``THRESHOLDS`` is the grid 0.05, 0.06, ... 0.95. Keep every cutoff whose
    phishing precision is at least ``min_precision``. Among those, take the
    highest recall. If two cutoffs tie on recall, take the higher cutoff,
    which flags fewer legitimate URLs.

    If no cutoff holds the precision floor, fall back to the highest F1
    and break ties the same way. The returned ``Metrics`` are for the
    labels and probabilities passed in, which the comparison script makes
    the validation slice, not the final test set.
    """
    candidates = [
        metrics_from_probabilities(labels, probabilities, threshold, n_train, model_name)
        for threshold in THRESHOLDS
    ]
    qualified = [metrics for metrics in candidates if metrics.precision >= min_precision]
    pool = qualified or candidates

    def sort_key(metrics: Metrics) -> tuple[float, float]:
        score = metrics.recall if qualified else metrics.f1
        return (score, metrics.threshold)

    return max(pool, key=sort_key)
