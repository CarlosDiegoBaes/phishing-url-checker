"""The result of scoring one URL, and the numbers from a held-out test."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """How strongly the checker leans toward phishing.

    The three bands are a display choice. The measured precision and recall
    use a single cutoff, the decision threshold, rather than these names.
    """

    LIKELY_SAFE = "LIKELY SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    LIKELY_PHISHING = "LIKELY PHISHING"


@dataclass(frozen=True)
class Verdict:
    """One URL's score and the notes that explain it.

    ``probability`` is the model's estimate of P(phishing), from 0 to 1.
    ``level`` is the band that probability falls into.
    ``reasons`` are short phrases for a person reading the result. They mix
    model contributions (when the model can explain itself) with rule flags
    that the model does not learn.
    """

    url: str
    probability: float
    level: RiskLevel
    reasons: tuple[str, ...] = ()

    @staticmethod
    def level_for(probability: float, threshold: float = 0.5) -> RiskLevel:
        """Map a probability onto a risk band.

        Below the decision threshold the URL is LIKELY SAFE.
        At or above 0.8, and still at or above the threshold, it is
        LIKELY PHISHING. The band between those two cuts is SUSPICIOUS.

        With the default threshold of 0.5 this is the original checker:
        0.5 starts SUSPICIOUS and 0.8 starts LIKELY PHISHING.
        """
        if probability >= threshold and probability >= 0.8:
            return RiskLevel.LIKELY_PHISHING
        if probability >= threshold:
            return RiskLevel.SUSPICIOUS
        return RiskLevel.LIKELY_SAFE


@dataclass(frozen=True)
class Metrics:
    """Held-out scores for one trained model.

    ``confusion`` has rows for the actual class and columns for the predicted
    class, in the order legitimate then phishing. So the four cells are
    true negatives, false positives, false negatives, true positives.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: list[list[int]]
    n_train: int
    n_test: int
    threshold: float = 0.5
    model_name: str = "logistic"

    def as_dict(self) -> dict[str, object]:
        """JSON-ready copy of the metrics."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion": self.confusion,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "threshold": self.threshold,
            "model": self.model_name,
        }
