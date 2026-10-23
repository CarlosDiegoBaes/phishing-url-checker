"""Lexical features and defanging, checked against hand-computed values."""

from __future__ import annotations

import pytest

from phishguard.features import (
    FeatureExtractor,
    LexicalExtractor,
    is_ip_address,
    refang,
    shannon_entropy,
)


def values(extractor: LexicalExtractor, url: str) -> dict[str, float]:
    """Map feature names to the numbers ``extract`` returned."""
    return dict(zip(extractor.names, extractor.extract(url), strict=True))


def test_feature_extractor_cannot_be_constructed() -> None:
    with pytest.raises(TypeError):
        FeatureExtractor()


def test_names_match_vector_length(extractor: LexicalExtractor) -> None:
    vector = extractor.extract("https://www.example.com")
    assert len(vector) == len(extractor.names) == 10
    assert len(set(extractor.names)) == 10


def test_refang_repairs_defanged_urls() -> None:
    assert refang("hxxps://bit[.]ly/3xYz") == "https://bit.ly/3xYz"
    assert refang("hxxp://evil[.]tk") == "http://evil.tk"


def test_www_prefix_is_ignored_for_length(extractor: LexicalExtractor) -> None:
    with_www = values(extractor, "https://www.ab.com")
    without_www = values(extractor, "https://ab.com")
    assert with_www["host_length"] == without_www["host_length"] == 6


def test_simple_host_features_are_exact(extractor: LexicalExtractor) -> None:
    row = values(extractor, "http://ab")
    assert row["host_length"] == 2
    assert row["num_dots"] == 0
    assert row["num_hyphens"] == 0
    assert row["has_ip_host"] == 0
    assert row["subdomain_depth"] == 0
    assert row["keyword_hits"] == 0
    assert row["host_entropy"] == 1.0
    assert row["risky_tld"] == 0
    assert row["digit_ratio"] == 0
    assert row["is_shortener"] == 0


def test_entropy_of_an_empty_host_is_zero() -> None:
    assert shannon_entropy("") == 0.0


def test_ip_host_digit_ratio_and_path_keyword(extractor: LexicalExtractor) -> None:
    row = values(extractor, "http://192.168.0.1/login")
    assert row["has_ip_host"] == 1
    assert row["host_length"] == 11
    assert row["num_dots"] == 3
    assert row["subdomain_depth"] == 2
    assert row["keyword_hits"] == 0
    assert row["digit_ratio"] == pytest.approx(8 / 11)
    assert is_ip_address("192.168.0.1")
    assert extractor.rule_flags("http://192.168.0.1/login") == [
        "no HTTPS",
        "credential keyword in URL",
    ]


def test_risky_tld_keywords_and_hyphens(extractor: LexicalExtractor) -> None:
    row = values(extractor, "HTTP://login-account.TK/reset")
    assert row["risky_tld"] == 1
    assert row["keyword_hits"] == 2
    assert row["num_hyphens"] == 1
    assert row["subdomain_depth"] == max(row["num_dots"] - 1, 0)


def test_known_shortener(extractor: LexicalExtractor) -> None:
    row = values(extractor, "https://bit.ly/abc")
    assert row["is_shortener"] == 1
    assert row["host_length"] == 6
    assert extractor.rule_flags("https://bit.ly/abc") == []


def test_at_sign_flag_uses_the_original_wording(extractor: LexicalExtractor) -> None:
    flags = extractor.rule_flags("http://user@example.com")
    assert flags[0] == "'@' in URL (browser ignores everything before it)"
    assert "no HTTPS" in flags


def test_transform_shape_and_empty_input(extractor: LexicalExtractor) -> None:
    rows = extractor.transform(["https://a.com", "https://b.com"])
    assert rows.shape == (2, 10)
    assert extractor.transform([]).shape == (0, 10)
