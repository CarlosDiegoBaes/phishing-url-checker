"""Phishing URL checker using a simple, interpretable ML model.

Extracts lexical features from a URL (no page is ever fetched) and scores it
with a scikit-learn LogisticRegression trained on the PhiUSIIL dataset.
"""
from __future__ import annotations

import argparse
import ipaddress
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).with_name("model.joblib")


class Verdict(Enum):
    LIKELY_SAFE = "LIKELY SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    LIKELY_PHISHING = "LIKELY PHISHING"


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    confusion: list[list[int]]  # rows = actual [legit, phish], cols = predicted
    n_train: int
    n_test: int


@dataclass(frozen=True)
class URLReport:
    url: str
    probability: float
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)


class URLFeatureExtractor:
    """Turns a URL string into a fixed-length numeric feature vector."""

    KEYWORDS: tuple[str, ...] = ("login", "verify", "secure", "account", "update",
                                 "signin", "bank", "confirm", "password", "wallet")
    SHORTENERS: frozenset[str] = frozenset({"bit.ly", "tinyurl.com", "goo.gl", "t.co",
                                            "ow.ly", "is.gd", "buff.ly", "cutt.ly", "rb.gy"})
    RISKY_TLDS: frozenset[str] = frozenset({"tk", "ml", "ga", "cf", "gq", "xyz", "top",
                                            "zip", "ru", "cn", "work", "click"})
    NAMES: tuple[str, ...] = ("host_length", "num_dots", "num_hyphens", "has_ip_host",
                              "subdomain_depth", "keyword_hits", "host_entropy",
                              "risky_tld", "digit_ratio", "is_shortener")

    @staticmethod
    def _host(url: str) -> str:
        if "://" not in url:
            url = "http://" + url
        return (urlparse(url).hostname or "").lower()

    @staticmethod
    def _entropy(text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        return -sum(c / len(text) * math.log2(c / len(text)) for c in counts.values())

    @staticmethod
    def _is_ip(host: str) -> bool:
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return False

    def extract(self, url: str) -> list[float]:
        """Model features use the hostname only (see README: dataset bias)."""
        host = self._host(url)
        bare = host[4:] if host.startswith("www.") else host
        tld = host.rsplit(".", 1)[-1] if "." in host else ""
        return [
            float(len(bare)),
            float(bare.count(".")),
            float(bare.count("-")),
            float(self._is_ip(host)),
            float(max(bare.count(".") - 1, 0)),
            float(sum(k in bare for k in self.KEYWORDS)),
            self._entropy(bare),
            float(tld in self.RISKY_TLDS),
            sum(ch.isdigit() for ch in bare) / max(len(bare), 1),
            float(bare in self.SHORTENERS),
        ]

    def rule_flags(self, url: str) -> list[str]:
        """Full-URL red flags reported alongside the model score (not learned)."""
        lower = url.lower()
        flags = []
        if "@" in url:
            flags.append("'@' in URL (browser ignores everything before it)")
        if not lower.startswith("https://"):
            flags.append("no HTTPS")
        if any(k in lower for k in self.KEYWORDS):
            flags.append("credential keyword in URL")
        return flags

    def transform(self, urls: list[str]) -> np.ndarray:
        return np.array([self.extract(u) for u in urls])


class PhishingClassifier:
    """Standardized LogisticRegression: simple, and its weights are readable."""

    def __init__(self, extractor: URLFeatureExtractor | None = None) -> None:
        self.extractor = extractor or URLFeatureExtractor()
        self.pipeline = Pipeline([("scale", StandardScaler()),
                                  ("lr", LogisticRegression(max_iter=1000))])

    def train(self, urls: list[str], labels: list[int], seed: int = 42) -> Metrics:
        """labels: 1 = phishing, 0 = legitimate."""
        X = self.extractor.transform(urls)
        y = np.array(labels)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                  random_state=seed, stratify=y)
        self.pipeline.fit(X_tr, y_tr)
        pred = self.pipeline.predict(X_te)
        return Metrics(accuracy_score(y_te, pred), precision_score(y_te, pred),
                       recall_score(y_te, pred), confusion_matrix(y_te, pred).tolist(),
                       len(y_tr), len(y_te))

    def weights(self) -> list[tuple[str, float]]:
        coefs = self.pipeline.named_steps["lr"].coef_[0]
        return sorted(zip(self.extractor.NAMES, coefs), key=lambda p: abs(p[1]), reverse=True)

    def score(self, url: str) -> URLReport:
        x = np.array([self.extractor.extract(url)])
        prob = float(self.pipeline.predict_proba(x)[0][1])
        scaled = self.pipeline.named_steps["scale"].transform(x)[0]
        contrib = scaled * self.pipeline.named_steps["lr"].coef_[0]
        reasons = [f"{name} (+{c:.2f})" for c, name in
                   sorted(zip(contrib, self.extractor.NAMES), reverse=True)[:3] if c > 0.3]
        reasons += self.extractor.rule_flags(url)
        verdict = (Verdict.LIKELY_PHISHING if prob >= 0.8 else
                   Verdict.SUSPICIOUS if prob >= 0.5 else Verdict.LIKELY_SAFE)
        return URLReport(url, prob, verdict, reasons)

    def save(self, path: Path = MODEL_PATH) -> None:
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> PhishingClassifier:
        clf = cls()
        clf.pipeline = joblib.load(path)
        return clf


def refang(url: str) -> str:
    return url.replace("hxxp", "http").replace("[.]", ".")


def print_report(r: URLReport) -> None:
    print(f"{r.url}\n  verdict: {r.verdict.value}   p(phish) = {r.probability:.2f}")
    for reason in r.reasons:
        print(f"  - {reason}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score URLs for phishing risk.")
    ap.add_argument("urls", nargs="*", help="URLs to score (defanged hxxp/[.] is fine)")
    ap.add_argument("--weights", action="store_true", help="print feature weights")
    args = ap.parse_args()
    if not MODEL_PATH.exists():
        sys.exit("model.joblib not found; run train.py first")
    clf = PhishingClassifier.load()
    if args.weights:
        for name, w in clf.weights():
            print(f"{name:>16}  {w:+.2f}")
    for url in args.urls:
        report = clf.score(refang(url))
        print_report(URLReport(url, report.probability, report.verdict, report.reasons))


if __name__ == "__main__":
    main()
