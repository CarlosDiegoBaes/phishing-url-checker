"""Turn a URL string into numbers the model can learn from.

A feature extractor is a class with one job: look at a URL and return a
fixed list of numbers. The abstract base class defines that job. Concrete
subclasses decide which signals to use. The rest of the program talks to
the base class, so a new extractor can be added without rewriting the model.
"""

from __future__ import annotations

import ipaddress
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from urllib.parse import urlparse

import numpy as np


def refang(url: str) -> str:
    """Turn a defanged URL back into text ``urlparse`` can read.

    People write ``hxxp``, ``[.]``, ``(.)``, and ``[:]`` so a suspicious link
    is not clicked by accident. ``hxxps`` becomes ``https`` because the
    letters ``hxxp`` are replaced first. Any mix of case, such as ``HXXPS``,
    comes back as lowercase ``https``.
    """
    url = re.sub("hxxp(s?)", lambda match: "http" + match.group(1).lower(), url, flags=re.I)
    return url.replace("[.]", ".").replace("(.)", ".").replace("[:]", ":")


def hostname_of(url: str) -> str:
    """Lowercase hostname, with ``http://`` added when the URL has no scheme."""
    if "://" not in url:
        url = "http://" + url
    return (urlparse(url).hostname or "").lower()


def shannon_entropy(text: str) -> float:
    """Shannon entropy in bits per character. Empty text is 0."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum(count / length * math.log2(count / length) for count in counts.values())


def is_ip_address(host: str) -> bool:
    """True when the hostname is an IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


class FeatureExtractor(ABC):
    """The interface every feature extractor implements.

    ``extract`` handles one URL. ``transform`` is shared: it calls
    ``extract`` for each URL and stacks the rows. Subclasses do not
    need to reimplement ``transform`` unless they have a faster way.
    """

    @property
    @abstractmethod
    def names(self) -> tuple[str, ...]:
        """Feature names, in the same order as ``extract``."""

    @abstractmethod
    def extract(self, url: str) -> list[float]:
        """Return one feature vector for a single URL."""

    def transform(self, urls: Sequence[str]) -> np.ndarray:
        """Stack ``extract`` across many URLs into a 2-D array of shape (n, features)."""
        if not urls:
            return np.empty((0, len(self.names)), dtype=float)
        return np.asarray([self.extract(url) for url in urls], dtype=float)

    def rule_flags(self, url: str) -> list[str]:
        """Extra red flags to print next to a score.

        These notes are for a person reading the verdict. They are not
        model features. The default is no flags.
        """
        return []


class LexicalExtractor(FeatureExtractor):
    """Hostname-only lexical features.

    In the PhiUSIIL dataset every legitimate URL is an ``https://`` homepage
    with no path. A model that sees path length or the scheme can score
    very high by learning "has a path" or "uses http", which is not the
    same as detecting phishing. This extractor therefore reads the hostname
    only. Full-URL observations are reported separately by ``rule_flags``.
    """

    KEYWORDS: tuple[str, ...] = (
        "login",
        "verify",
        "secure",
        "account",
        "update",
        "signin",
        "bank",
        "confirm",
        "password",
        "wallet",
    )
    SHORTENERS: frozenset[str] = frozenset(
        {
            "bit.ly",
            "tinyurl.com",
            "goo.gl",
            "t.co",
            "ow.ly",
            "is.gd",
            "buff.ly",
            "cutt.ly",
            "rb.gy",
        }
    )
    RISKY_TLDS: frozenset[str] = frozenset(
        {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "zip", "ru", "cn", "work", "click"}
    )
    _NAMES: tuple[str, ...] = (
        "host_length",
        "num_dots",
        "num_hyphens",
        "has_ip_host",
        "subdomain_depth",
        "keyword_hits",
        "host_entropy",
        "risky_tld",
        "digit_ratio",
        "is_shortener",
    )

    @property
    def names(self) -> tuple[str, ...]:
        """The ten hostname features, in vector order."""
        return self._NAMES

    def extract(self, url: str) -> list[float]:
        """Build the hostname feature vector for one URL.

        A leading ``www.`` is stripped before length, dots, hyphens, keywords,
        entropy, and the shortener check. The IP check still sees the raw host.
        """
        host = hostname_of(url)
        bare = host[4:] if host.startswith("www.") else host
        tld = host.rsplit(".", 1)[-1] if "." in host else ""
        keyword_hits = sum(keyword in bare for keyword in self.KEYWORDS)
        digits = sum(character.isdigit() for character in bare)
        return [
            float(len(bare)),
            float(bare.count(".")),
            float(bare.count("-")),
            float(is_ip_address(host)),
            float(max(bare.count(".") - 1, 0)),
            float(keyword_hits),
            shannon_entropy(bare),
            float(tld in self.RISKY_TLDS),
            digits / max(len(bare), 1),
            float(bare in self.SHORTENERS),
        ]

    def rule_flags(self, url: str) -> list[str]:
        """Full-URL red flags shown beside the score.

        The same dataset bias that keeps these out of the model is why they
        are printed as notes instead: "@" in the URL, a missing HTTPS scheme,
        and a credential keyword anywhere in the URL.
        """
        lowered = url.lower()
        flags: list[str] = []
        if "@" in url:
            flags.append("'@' in URL (browser ignores everything before it)")
        if not lowered.startswith("https://"):
            flags.append("no HTTPS")
        if any(keyword in lowered for keyword in self.KEYWORDS):
            flags.append("credential keyword in URL")
        return flags
