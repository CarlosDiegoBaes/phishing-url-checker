"""Compose feature extractors with a model and turn scores into verdicts.

This is the object the command line and the trainer both talk to. It owns
the extractors and the model, so either piece can be replaced on its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np

from phishguard.features import FeatureExtractor, LexicalExtractor
from phishguard.models import LogisticModel, Model
from phishguard.verdict import Verdict


class Pipeline:
    """Extract features, score them, and package the result as a ``Verdict``.

    Pass more than one extractor to concatenate their columns. The shipped
    checker uses a single ``LexicalExtractor`` and a ``LogisticModel``.
    """

    def __init__(
        self,
        extractors: Sequence[FeatureExtractor] | None = None,
        model: Model | None = None,
        threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if extractors is not None and len(extractors) == 0:
            raise ValueError("pipeline needs at least one feature extractor")
        self.extractors: tuple[FeatureExtractor, ...] = (
            tuple(extractors) if extractors is not None else (LexicalExtractor(),)
        )
        self.model: Model = model if model is not None else LogisticModel()
        self.threshold = threshold

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of every column the extractors produce, in order."""
        return tuple(name for extractor in self.extractors for name in extractor.names)

    def transform(self, urls: Sequence[str]) -> np.ndarray:
        """Run every extractor and concatenate the columns."""
        blocks = [extractor.transform(urls) for extractor in self.extractors]
        return np.hstack(blocks)

    def fit(self, urls: Sequence[str], labels: Sequence[int]) -> None:
        """Learn from raw URLs. ``1`` is phishing and ``0`` is legitimate."""
        features = self.transform(urls)
        self.model.fit(features, np.asarray(labels))

    def score(self, url: str) -> Verdict:
        """Score one URL. The URL should already be refanged."""
        features = self.transform([url])
        probability = float(self.model.predict_proba(features)[0])
        reasons = tuple(self._reasons(url, features))
        level = Verdict.level_for(probability, self.threshold)
        return Verdict(url, probability, level, reasons)

    def feature_weights(self) -> list[tuple[str, float]]:
        """Feature weights sorted by absolute value, largest first.

        An empty list means this model has no single weight per feature.
        """
        coefficients = self.model.coefficients()
        if coefficients is None:
            return []
        names = self.feature_names
        if len(coefficients) != len(names):
            raise RuntimeError("model weights do not line up with feature names")
        pairs = [(name, float(weight)) for name, weight in zip(names, coefficients, strict=True)]
        return sorted(pairs, key=lambda pair: abs(pair[1]), reverse=True)

    def save(self, path: Path) -> None:
        """Write the extractors, the model, and the threshold to disk."""
        joblib.dump(
            {
                "version": 1,
                "threshold": self.threshold,
                "extractors": self.extractors,
                "model": self.model,
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: Path) -> Pipeline:
        """Load a pipeline written by ``save``."""
        payload = joblib.load(Path(path))
        return cls(
            extractors=payload["extractors"],
            model=payload["model"],
            threshold=payload["threshold"],
        )

    def _reasons(self, url: str, features: np.ndarray) -> list[str]:
        """Top feature contributions, then any rule flags.

        Contributions use the same rule as the original checker: look at the
        three largest, and keep the ones that push toward phishing by more
        than 0.3 on the log-odds scale.
        """
        reasons: list[str] = []
        contributions = self.model.explain(features)
        if contributions is not None:
            ranked = sorted(
                zip(contributions[0].tolist(), self.feature_names, strict=True),
                reverse=True,
            )[:3]
            reasons.extend(f"{name} (+{value:.2f})" for value, name in ranked if value > 0.3)
        for extractor in self.extractors:
            reasons.extend(extractor.rule_flags(url))
        return reasons
