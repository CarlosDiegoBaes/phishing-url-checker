"""Command-line scoring against a tiny model saved in a temp directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from phishguard.cli import format_verdict, main
from phishguard.models import LogisticModel
from phishguard.pipeline import Pipeline
from phishguard.verdict import RiskLevel, Verdict


def _save_tiny_model(path: Path, urls: list[str], labels: list[int]) -> Pipeline:
    pipeline = Pipeline(model=LogisticModel())
    pipeline.fit(urls, labels)
    pipeline.save(path)
    return pipeline


def test_format_matches_the_original_layout() -> None:
    text = format_verdict(Verdict("https://www.example.com", 0.174, RiskLevel.LIKELY_SAFE, ()))
    assert text == "https://www.example.com\n  verdict: LIKELY SAFE   p(phish) = 0.17"


def test_cli_prints_the_original_url_and_a_verdict(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    training_urls: list[str],
    training_labels: list[int],
) -> None:
    path = tmp_path / "model.joblib"
    _save_tiny_model(path, training_urls, training_labels)
    main(["--model", str(path), "hxxp://paypal-account-verify[.]secure-login[.]tk/signin"])
    output = capsys.readouterr().out
    assert output.startswith("hxxp://paypal-account-verify[.]secure-login[.]tk/signin\n")
    assert "LIKELY PHISHING" in output
    assert "p(phish)" in output
    assert "no HTTPS" in output


def test_cli_prints_weights(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    training_urls: list[str],
    training_labels: list[int],
) -> None:
    path = tmp_path / "model.joblib"
    _save_tiny_model(path, training_urls, training_labels)
    main(["--model", str(path), "--weights"])
    output = capsys.readouterr().out
    assert "host_length" in output
    assert "digit_ratio" in output


def test_cli_missing_model_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--model", str(tmp_path / "missing.joblib")])
    assert caught.value.code != 0
