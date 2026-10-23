"""Pipeline composition, verdict bands, explanations, and save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phishguard.features import FeatureExtractor, LexicalExtractor
from phishguard.models import LogisticModel, Model
from phishguard.pipeline import Pipeline
from phishguard.verdict import RiskLevel


class CountingExtractor(FeatureExtractor):
    """Test double that adds one constant column and one flag."""

    @property
    def names(self) -> tuple[str, ...]:
        return ("ones",)

    def extract(self, url: str) -> list[float]:
        return [1.0]

    def rule_flags(self, url: str) -> list[str]:
        return ["from the extra extractor"]


class FakeModel(Model):
    """A model with a fixed probability and a known contribution vector."""

    name = "fake"

    def _fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._width = features.shape[1]

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), 0.91)

    def explain(self, features: np.ndarray) -> np.ndarray:
        row = np.zeros(features.shape[1])
        row[0] = 1.234
        row[1] = 0.1
        return np.tile(row, (len(features), 1))

    def coefficients(self) -> np.ndarray:
        weights = np.zeros(self._width)
        weights[0] = -1.5
        weights[1] = 0.25
        return weights


def test_model_cannot_be_constructed() -> None:
    with pytest.raises(TypeError):
        Model()


def test_scoring_before_fit_raises() -> None:
    pipeline = Pipeline()
    with pytest.raises(RuntimeError):
        pipeline.score("https://www.example.com")


def test_threshold_must_be_a_probability() -> None:
    with pytest.raises(ValueError):
        Pipeline(threshold=1.5)
    with pytest.raises(ValueError):
        Pipeline(extractors=[])


def test_extractors_are_concatenated() -> None:
    pipeline = Pipeline(extractors=(LexicalExtractor(), CountingExtractor()), model=FakeModel())
    pipeline.fit(["https://www.example.com"], [0])
    assert pipeline.feature_names[-1] == "ones"
    assert len(pipeline.feature_names) == 11
    verdict = pipeline.score("https://www.example.com")
    assert "from the extra extractor" in verdict.reasons


def test_explanation_keeps_the_top_positive_contributions() -> None:
    pipeline = Pipeline(model=FakeModel())
    pipeline.fit(["https://www.example.com"], [0])
    verdict = pipeline.score("https://www.example.com")
    assert verdict.level is RiskLevel.LIKELY_PHISHING
    assert verdict.probability == pytest.approx(0.91)
    assert verdict.reasons == ("host_length (+1.23)",)


def test_weights_are_sorted_by_absolute_value() -> None:
    pipeline = Pipeline(model=FakeModel())
    pipeline.fit(["https://www.example.com"], [0])
    weights = pipeline.feature_weights()
    assert weights[0] == ("host_length", pytest.approx(-1.5))
    assert abs(weights[0][1]) >= abs(weights[1][1])


def test_level_bands_follow_the_threshold() -> None:
    from phishguard.verdict import Verdict

    assert Verdict.level_for(0.49) is RiskLevel.LIKELY_SAFE
    assert Verdict.level_for(0.50) is RiskLevel.SUSPICIOUS
    assert Verdict.level_for(0.79) is RiskLevel.SUSPICIOUS
    assert Verdict.level_for(0.80) is RiskLevel.LIKELY_PHISHING
    assert Verdict.level_for(0.60, threshold=0.35) is RiskLevel.SUSPICIOUS
    assert Verdict.level_for(0.90, threshold=0.95) is RiskLevel.LIKELY_SAFE
    assert Verdict.level_for(0.96, threshold=0.95) is RiskLevel.LIKELY_PHISHING


def test_logistic_separates_obvious_phishing_from_homepages(
    training_urls: list[str],
    training_labels: list[int],
) -> None:
    pipeline = Pipeline(model=LogisticModel())
    pipeline.fit(training_urls, training_labels)
    phish = pipeline.score("http://paypal-account-verify.secure-login.tk/signin")
    safe = pipeline.score("https://www.example.com")
    assert phish.probability > safe.probability
    assert phish.probability > 0.7
    assert safe.probability < 0.3
    assert any("(+" in reason for reason in phish.reasons)
    assert "no HTTPS" in phish.reasons
    assert "credential keyword in URL" in phish.reasons


def test_save_and_load_round_trip(
    tmp_path: Path, training_urls: list[str], training_labels: list[int]
) -> None:
    pipeline = Pipeline(model=LogisticModel(), threshold=0.5)
    pipeline.fit(training_urls, training_labels)
    path = tmp_path / "model.joblib"
    pipeline.save(path)
    loaded = Pipeline.load(path)
    url = "http://login-verify-account.update-secure.xyz/password"
    assert loaded.score(url).probability == pytest.approx(pipeline.score(url).probability)
    assert loaded.threshold == 0.5
    assert loaded.feature_weights()[0][0] == pipeline.feature_weights()[0][0]
