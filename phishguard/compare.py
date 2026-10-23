"""Compare logistic regression, a random forest, and gradient boosting.

The held-out test rows are the same 20% split (seed 42) as the baseline.
Thresholds are chosen on a validation slice cut from the training rows, so
the test set is not used to pick the cutoff. Each model is then refit on
the full training split and scored once on the test set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from phishguard.evaluate import (
    MIN_PRECISION,
    choose_threshold,
    metrics_from_probabilities,
    split_features,
)
from phishguard.models import Model
from phishguard.registry import MODEL_ORDER, ModelRegistry, default_registry
from phishguard.verdict import Metrics

BASELINE_CONFUSION = [[24865, 2105], [7658, 12531]]


@dataclass(frozen=True)
class ModelComparison:
    """Validation choice and test scores for one model."""

    name: str
    test_at_half: Metrics
    validation: Metrics
    test_at_chosen_threshold: Metrics

    def as_dict(self) -> dict[str, object]:
        """JSON-ready copy. Probabilities are not included."""
        return {
            "test_at_0.5": self.test_at_half.as_dict(),
            "chosen_threshold": self.validation.threshold,
            "validation_at_chosen_threshold": self.validation.as_dict(),
            "test_at_chosen_threshold": self.test_at_chosen_threshold.as_dict(),
        }


@dataclass(frozen=True)
class ComparisonResult:
    """Every model, which one was selected, and the test-set probabilities."""

    models: tuple[ModelComparison, ...]
    winner: str
    test_labels: np.ndarray
    test_probabilities: dict[str, np.ndarray]
    fitted_models: dict[str, Model]
    n_fit: int
    n_validation: int

    def winner_comparison(self) -> ModelComparison:
        """The selected model's report."""
        for model in self.models:
            if model.name == self.winner:
                return model
        raise RuntimeError(f"winner {self.winner!r} is missing from the reports")


def select_winner(
    validation: dict[str, Metrics],
    order: tuple[str, ...] = MODEL_ORDER,
    min_precision: float = MIN_PRECISION,
) -> str:
    """Choose a model from validation scores only.

    Prefer models whose chosen threshold still has precision at least
    ``min_precision``. Among that pool, take the highest recall, then the
    highest precision, then the earlier name in ``order`` (logistic, then
    the forest, then boosting). If nobody meets the precision floor, the
    same ranking is applied to every model.
    """
    eligible = [name for name in order if validation[name].precision + 1e-12 >= min_precision]
    pool = eligible or list(order)

    def sort_key(name: str) -> tuple[float, float, int]:
        metrics = validation[name]
        return (-metrics.recall, -metrics.precision, order.index(name))

    return min(pool, key=sort_key)


def compare_models(
    features: np.ndarray,
    labels: np.ndarray,
    registry: ModelRegistry | None = None,
) -> ComparisonResult:
    """Fit every registered model and score the shared holdout.

    Logistic regression at threshold 0.5 must reproduce the published
    confusion matrix. That check guards the split and the features.
    """
    registry = registry or default_registry()
    x_train, x_test, y_train, y_test = split_features(features, labels)
    x_fit, x_val, y_fit, y_val = split_features(x_train, y_train)
    reports: list[ModelComparison] = []
    validation: dict[str, Metrics] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    fitted: dict[str, Model] = {}

    for name in registry.names():
        print(f"fitting {name} on the validation split...", flush=True)
        scout = registry.create(name)
        scout.fit(x_fit, y_fit)
        validation_choice = choose_threshold(
            y_val,
            scout.predict_proba(x_val),
            n_train=len(y_fit),
            model_name=name,
        )
        print(
            f"  validation threshold {validation_choice.threshold:.2f} "
            f"precision {validation_choice.precision:.4f} "
            f"recall {validation_choice.recall:.4f}",
            flush=True,
        )

        print(f"refitting {name} on the full training split...", flush=True)
        model = registry.create(name)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        at_half = metrics_from_probabilities(
            y_test, probabilities, 0.5, n_train=len(y_train), model_name=name
        )
        tuned = metrics_from_probabilities(
            y_test,
            probabilities,
            validation_choice.threshold,
            n_train=len(y_train),
            model_name=name,
        )
        if name == "logistic" and at_half.confusion != BASELINE_CONFUSION:
            raise RuntimeError(
                "Logistic regression at 0.5 did not reproduce the baseline "
                f"confusion matrix: {at_half.confusion}"
            )
        reports.append(
            ModelComparison(
                name=name,
                test_at_half=at_half,
                validation=validation_choice,
                test_at_chosen_threshold=tuned,
            )
        )
        validation[name] = validation_choice
        test_probabilities[name] = probabilities
        fitted[name] = model

    winner = select_winner(validation)
    return ComparisonResult(
        models=tuple(reports),
        winner=winner,
        test_labels=y_test,
        test_probabilities=test_probabilities,
        fitted_models=fitted,
        n_fit=int(len(y_fit)),
        n_validation=int(len(y_val)),
    )


def comparison_payload(result: ComparisonResult) -> dict[str, object]:
    """The measured comparison, without the raw probability arrays."""
    winner = result.winner_comparison()
    return {
        "split": {
            "seed": 42,
            "test_size": 0.2,
            "validation_fraction_of_train": 0.2,
            "n_fit": result.n_fit,
            "n_validation": result.n_validation,
            "n_train": winner.test_at_half.n_train,
            "n_test": winner.test_at_half.n_test,
        },
        "rule": (
            "On the validation slice, choose the threshold from 0.05 to 0.95 "
            "with the highest phishing recall whose precision is at least 0.80. "
            "Ties go to the higher threshold. If none qualify, use the highest F1. "
            "Then pick the model with the highest validation recall among those "
            "that still meet that precision floor. The test set is scored after "
            "that choice, on a model refit on the full training split."
        ),
        "min_precision": MIN_PRECISION,
        "models": {report.name: report.as_dict() for report in result.models},
        "winner": result.winner,
        "winner_threshold": winner.validation.threshold,
    }


def write_comparison(path: str, result: ComparisonResult) -> None:
    """Write ``comparison.json``."""
    payload = comparison_payload(result)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
