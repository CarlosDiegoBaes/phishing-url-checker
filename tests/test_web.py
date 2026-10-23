"""The web app, driven through FastAPI's TestClient.

Most tests load the shipped ``model.joblib`` read-only, so they check the
real forest and its saved threshold. One test uses a tiny logistic model to
cover the branch where a model can explain its score.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from phishguard.config import DEFAULT_MODEL, DEFAULT_THRESHOLD
from phishguard.models import LogisticModel
from phishguard.paths import MODEL_PATH
from phishguard.pipeline import Pipeline
from phishguard.web import create_app
from phishguard.web.service import IP_HOST_FLAG, CheckService

DEFANGED = "hxxp://paypal-account-verify[.]secure-login[.]tk/signin"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A client for the app built around the shipped model."""
    return TestClient(create_app(MODEL_PATH))


def test_check_uses_the_shipped_model_and_threshold(client: TestClient) -> None:
    body = client.post("/check", json={"url": "https://www.example.com"}).json()
    assert body["model"] == DEFAULT_MODEL
    assert body["threshold"] == DEFAULT_THRESHOLD
    assert 0.0 <= body["score"] <= 1.0
    assert body["band"] in {"LIKELY SAFE", "SUSPICIOUS", "LIKELY PHISHING"}
    assert body["verdict"] in {"phishing", "legitimate"}


def test_score_matches_the_pipeline_directly(client: TestClient) -> None:
    url = "http://login-verify-account.update-secure.xyz/password"
    expected = Pipeline.load(MODEL_PATH).score(url)
    body = client.post("/check", json={"url": url}).json()
    assert body["score"] == pytest.approx(expected.probability)
    assert body["band"] == expected.level.value


@pytest.mark.parametrize(
    "defanged",
    [
        DEFANGED,
        "hxxp[:]//paypal-account-verify(.)secure-login(.)tk/signin",
        "HXXP://paypal-account-verify[.]secure-login[.]tk/signin",
    ],
)
def test_defanged_input_is_refanged(client: TestClient, defanged: str) -> None:
    body = client.post("/check", json={"url": defanged}).json()
    assert body["input"] == defanged
    assert body["url"] == "http://paypal-account-verify.secure-login.tk/signin"
    plain = client.post("/check", json={"url": body["url"]}).json()
    assert body["score"] == pytest.approx(plain["score"])


def test_obvious_phishing_url_is_flagged(client: TestClient) -> None:
    body = client.post("/check", json={"url": DEFANGED}).json()
    assert body["verdict"] == "phishing"
    assert body["band"] in {"SUSPICIOUS", "LIKELY PHISHING"}
    assert body["score"] >= DEFAULT_THRESHOLD
    assert "no HTTPS" in body["rule_flags"]
    assert "credential keyword in URL" in body["rule_flags"]


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.0.1/login",
        "https://203.0.113.7:8443/",
        "http://[2001:db8::1]/",
        "10[.]0[.]0[.]1",
    ],
)
def test_ip_address_host_gets_a_plain_english_flag(client: TestClient, url: str) -> None:
    body = client.post("/check", json={"url": url}).json()
    assert IP_HOST_FLAG in body["rule_flags"]
    assert "raw IP address" in IP_HOST_FLAG


@pytest.mark.parametrize(
    "url", ["https://www.example.com", "https://1password.com", "http://192.168.0.1.evil.tk/"]
)
def test_named_host_gets_no_ip_flag(client: TestClient, url: str) -> None:
    body = client.post("/check", json={"url": url}).json()
    assert IP_HOST_FLAG not in body["rule_flags"]


def test_verdict_agrees_with_threshold(client: TestClient) -> None:
    for url in ["https://www.github.com", DEFANGED, "https://bit.ly/3xYz"]:
        body = client.post("/check", json={"url": url}).json()
        assert (body["verdict"] == "phishing") == (body["score"] >= body["threshold"])
        assert (body["band"] == "LIKELY SAFE") == (body["verdict"] == "legitimate")


def test_forest_reports_features_but_no_attributions(client: TestClient) -> None:
    body = client.post("/check", json={"url": "https://www.example.com"}).json()
    assert body["contributions"] is None
    names = [feature["name"] for feature in body["features"]]
    assert names == list(Pipeline.load(MODEL_PATH).feature_names)
    host_length = next(f for f in body["features"] if f["name"] == "host_length")
    assert host_length["value"] == len("example.com")
    assert "does not report how much" in body["explanation"]
    assert "None of the simple warning rules fired." in body["explanation"]


def test_surrounding_whitespace_is_ignored(client: TestClient) -> None:
    body = client.post("/check", json={"url": "  https://www.example.com \n"}).json()
    assert body["input"] == "https://www.example.com"


@pytest.mark.parametrize(
    "payload",
    [{}, {"url": ""}, {"url": "   "}, {"url": 42}, {"link": "x"}, {"url": "a" * 2049}],
)
def test_bad_input_is_rejected(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/check", json=payload).status_code == 422


def test_index_serves_the_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<form id="check-form"' in response.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_self_hosted_fonts_are_served(client: TestClient) -> None:
    for name in ["inter-latin-400-normal", "jetbrains-mono-latin-400-normal"]:
        response = client.get(f"/static/fonts/{name}.woff2")
        assert response.status_code == 200
        assert response.content[:4] == b"wOF2"


def test_healthz(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body == {"status": "ok", "model": DEFAULT_MODEL, "threshold": DEFAULT_THRESHOLD}


def test_missing_model_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train.py"):
        create_app(tmp_path / "missing.joblib")


def test_explainable_model_returns_contributions(
    training_urls: list[str], training_labels: list[int]
) -> None:
    pipeline = Pipeline(model=LogisticModel())
    pipeline.fit(training_urls, training_labels)
    client = TestClient(create_app(service=CheckService(pipeline)))
    body = client.post("/check", json={"url": DEFANGED}).json()
    contributions = body["contributions"]
    assert len(contributions) == len(pipeline.feature_names)
    magnitudes = [abs(item["value"]) for item in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert "pushed the score up the most" in body["explanation"]
