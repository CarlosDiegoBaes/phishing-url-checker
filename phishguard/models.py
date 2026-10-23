"""Classifiers that turn feature rows into a phishing probability.

``Model`` is the interface. It knows how to fit, how to score, and how to
say "I have no per-feature weights" by returning ``None``. ``LogisticModel``
is the concrete classifier: a standardized logistic regression, the same
one the original checker trained. A different algorithm can subclass
``Model`` and the pipeline does not change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler


class Model(ABC):
    """Strategy interface for a phishing classifier.

    ``fit`` is a template method. Subclasses implement ``_fit`` and
    ``_predict_proba``. The base class records that training happened,
    so scoring before ``fit`` fails with a clear error.
    """

    name: str = "model"

    def __init__(self) -> None:
        self._is_fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Learn from feature rows. ``1`` is phishing and ``0`` is legitimate."""
        self._fit(features, np.asarray(labels))
        self._is_fitted = True

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return P(phishing) for each row. Shape is ``(n,)``."""
        self._check_fitted()
        return self._predict_proba(features)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Label a row as phishing when its probability is at least ``threshold``."""
        return (self.predict_proba(features) >= threshold).astype(int)

    def explain(self, features: np.ndarray) -> np.ndarray | None:
        """Per-feature contributions, or ``None`` when this model cannot give them.

        When present, the array has shape ``(n, n_features)``. A positive
        value pushed that row toward phishing.
        """
        return None

    def coefficients(self) -> np.ndarray | None:
        """One weight per feature, or ``None`` when the model has no such weights."""
        return None

    @abstractmethod
    def _fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the underlying estimator. Called by ``fit``."""

    @abstractmethod
    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return P(phishing) for each row. Called after the fitted check."""

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before scoring, or load a saved pipeline.")


class LogisticModel(Model):
    """Standardized logistic regression.

    Scaling puts every feature on a comparable range, so a weight of ``+0.6``
    means "this feature pushes toward phishing" and can be compared with the
    other weights. ``max_iter=1000`` matches the original trainer.
    """

    name = "logistic"

    def __init__(self) -> None:
        super().__init__()
        self._estimator = SklearnPipeline(
            [
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000)),
            ]
        )

    def explain(self, features: np.ndarray) -> np.ndarray:
        """Contribution of each feature on the log-odds scale.

        Each value is the standardized feature times its coefficient.
        """
        self._check_fitted()
        scaled = self._estimator.named_steps["scale"].transform(features)
        coefficients = self._estimator.named_steps["lr"].coef_[0]
        return scaled * coefficients

    def coefficients(self) -> np.ndarray:
        """Standardized logistic-regression weights, one per feature."""
        self._check_fitted()
        return np.asarray(self._estimator.named_steps["lr"].coef_[0], dtype=float)

    def _fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._estimator.fit(features, labels)

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self._estimator.predict_proba(features)[:, 1]
