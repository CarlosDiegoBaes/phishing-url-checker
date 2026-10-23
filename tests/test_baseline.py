"""Reproduce the published numbers when the CSV is present.

CI does not download PhiUSIIL. Pytest's default options skip this module.
Run it locally with ``pytest -m dataset``.
"""

from __future__ import annotations

import json

import pytest

from phishguard.config import DEFAULT_MODEL, DEFAULT_THRESHOLD
from phishguard.paths import DEFAULT_CSV, ROOT
from phishguard.registry import default_registry
from phishguard.train import fit_holdout, load_urls, train

pytestmark = pytest.mark.dataset

PUBLISHED_WEIGHTS = {
    "digit_ratio": 1.0876088242028081,
    "host_length": 0.7674212983864499,
    "risky_tld": 0.6452881804919618,
    "num_hyphens": 0.5719064504488268,
    "is_shortener": 0.38213000309850065,
    "num_dots": 0.3650321370762692,
    "subdomain_depth": 0.3650321370762692,
    "keyword_hits": 0.22760856033151178,
    "host_entropy": -0.18823419885745835,
    "has_ip_host": 0.02387478578445962,
}


@pytest.mark.skipif(not DEFAULT_CSV.exists(), reason="PhiUSIIL CSV is not on disk")
def test_logistic_baseline_matches_the_published_holdout() -> None:
    urls, labels = load_urls(DEFAULT_CSV)
    pipeline, metrics = train(urls, labels)
    assert metrics.n_train == 188636
    assert metrics.n_test == 47159
    assert metrics.confusion == [[24865, 2105], [7658, 12531]]
    assert metrics.accuracy == pytest.approx(0.7929769503170127)
    assert metrics.precision == pytest.approx(0.8561765509702104)
    assert metrics.recall == pytest.approx(0.6206845311803457)
    weights = dict(pipeline.feature_weights())
    for name, published in PUBLISHED_WEIGHTS.items():
        assert weights[name] == pytest.approx(published, rel=1e-6, abs=1e-6)


@pytest.mark.skipif(not DEFAULT_CSV.exists(), reason="PhiUSIIL CSV is not on disk")
def test_shipped_model_matches_the_comparison_file() -> None:
    """Refit the frozen model and compare it with comparison.json.

    The file is the record of the validation choice. This test checks that
    training that model again on the same split reproduces the test scores.
    """
    recorded = json.loads((ROOT / "comparison.json").read_text(encoding="utf-8"))
    urls, labels = load_urls(DEFAULT_CSV)
    model = default_registry().create(DEFAULT_MODEL)
    _pipeline, metrics = fit_holdout(urls, labels, model, DEFAULT_THRESHOLD)
    saved = recorded["models"][DEFAULT_MODEL]["test_at_chosen_threshold"]
    assert recorded["winner"] == DEFAULT_MODEL
    assert recorded["winner_threshold"] == DEFAULT_THRESHOLD
    assert metrics.confusion == saved["confusion"]
    assert metrics.accuracy == pytest.approx(saved["accuracy"])
    assert metrics.precision == pytest.approx(saved["precision"])
    assert metrics.recall == pytest.approx(saved["recall"])
    assert metrics.f1 == pytest.approx(saved["f1"])
