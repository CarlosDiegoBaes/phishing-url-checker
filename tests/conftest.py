"""Shared URL fixtures. These are synthetic, so CI never downloads PhiUSIIL."""

from __future__ import annotations

import pytest

from phishguard.features import LexicalExtractor

BENIGN_URLS = [
    "https://www.example.com",
    "https://www.github.com",
    "https://en.wikipedia.org",
    "https://www.python.org",
]

PHISHING_URLS = [
    "http://paypal-account-verify.secure-login.tk/signin",
    "http://login-verify-account.update-secure.xyz/password",
    "http://bank-wallet-confirm.account-update.top",
    "http://secure-signin-password.verify-account.gq",
]


@pytest.fixture
def extractor() -> LexicalExtractor:
    """A fresh lexical extractor."""
    return LexicalExtractor()


@pytest.fixture
def training_urls() -> list[str]:
    """A tiny balanced set the unit tests can fit in memory."""
    return (BENIGN_URLS + PHISHING_URLS) * 8


@pytest.fixture
def training_labels() -> list[int]:
    """Labels aligned with ``training_urls``. 1 is phishing."""
    return ([0] * len(BENIGN_URLS) + [1] * len(PHISHING_URLS)) * 8
