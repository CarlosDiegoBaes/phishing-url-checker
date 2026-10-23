"""Classifiers that turn feature rows into a phishing probability.

``Model`` is the interface. ``SklearnProbabilityModel`` is the shared
scikit-learn plumbing. ``LogisticModel`` adds readable weights.
``RandomForestModel`` and ``GradientBoostingModel`` are the other two
strategies the comparison trains. The pipeline does not care which one it holds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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


class SklearnProbabilityModel(Model):
    """A scikit-learn classifier that already knows ``fit`` and ``predict_proba``.

    Subclasses pass a configured estimator to ``__init__``. This class
    handles training and the phishing column of ``predict_proba``.
    """

    def __init__(self, estimator: Any) -> None:
        super().__init__()
        self._estimator = estimator

    def _fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._estimator.fit(features, labels)

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = self._estimator.predict_proba(features)
        return np.asarray(probabilities[:, 1], dtype=float)


class LogisticModel(SklearnProbabilityModel):
    """Standardized logistic regression.

    Scaling puts every feature on a comparable range, so a weight can be
    compared with the other weights. ``max_iter=1000`` matches the original
    trainer. These settings are the baseline, not a tuned search.
    """

    name = "logistic"

    def __init__(self) -> None:
        estimator = SklearnPipeline(
            [
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000)),
            ]
        )
        super().__init__(estimator)

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


class RandomForestModel(SklearnProbabilityModel):
    """A forest of decision trees.

    Two hundred trees, depth capped at 16, at least five rows per leaf.
    ``random_state=42`` keeps a repeat run on the same machine stable.
    These are straightforward settings, not the result of a grid search.
    The forest has no single signed weight per feature.
    """

    name = "random_forest"

    def __init__(self) -> None:
        estimator = RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        super().__init__(estimator)


class GradientBoostingModel(SklearnProbabilityModel):
    """Histogram gradient boosting (``HistGradientBoostingClassifier``).

    This is scikit-learn's gradient boosting model for larger tables.
    It builds 200 trees of depth at most 6, with learning rate 0.1.
    Early stopping is off, so training does not carve out a private
    validation slice on top of the one the comparison uses.
    ``random_state=42``. These settings were not grid-searched.
    """

    name = "gradient_boosting"

    def __init__(self) -> None:
        estimator = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=6,
            early_stopping=False,
            random_state=42,
        )
        super().__init__(estimator)
