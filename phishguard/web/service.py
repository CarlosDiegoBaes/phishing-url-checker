"""Score one URL for the web app and describe the result in plain English.

The web layer adds no new scoring logic. ``CheckService`` refangs the input,
asks the saved ``Pipeline`` for a verdict, and then lays out what the model
actually saw: the raw feature values and the rule flags. A random forest has
no per-feature weights, so the service only reports contributions when the
model itself can supply them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phishguard.features import hostname_of, is_ip_address, refang
from phishguard.paths import MODEL_PATH
from phishguard.pipeline import Pipeline
from phishguard.verdict import RiskLevel


@dataclass(frozen=True)
class FeatureInfo:
    """A readable label and a one-line description for one model feature."""

    label: str
    description: str


FEATURE_INFO: dict[str, FeatureInfo] = {
    "host_length": FeatureInfo(
        "Hostname length", "Characters in the hostname, not counting a leading www."
    ),
    "num_dots": FeatureInfo("Dots", "How many dots the hostname has."),
    "num_hyphens": FeatureInfo("Hyphens", "How many hyphens the hostname has."),
    "has_ip_host": FeatureInfo(
        "IP address host", "1 when the host is a raw IP address instead of a name."
    ),
    "subdomain_depth": FeatureInfo(
        "Subdomain depth", "Dots minus one. This is almost the same measurement as Dots."
    ),
    "keyword_hits": FeatureInfo(
        "Credential keywords",
        "How many words like login, verify, secure, or account appear in the hostname.",
    ),
    "host_entropy": FeatureInfo(
        "Hostname entropy",
        "How mixed the characters are, in bits per character. Random-looking names score higher.",
    ),
    "risky_tld": FeatureInfo(
        "Risky TLD", "1 when the ending (such as .tk or .xyz) is on a short list of risky TLDs."
    ),
    "digit_ratio": FeatureInfo("Digit ratio", "Share of hostname characters that are digits."),
    "is_shortener": FeatureInfo(
        "URL shortener", "1 when the hostname is a known link shortener such as bit.ly."
    ),
}


IP_HOST_FLAG = (
    "The link points to a raw IP address instead of a domain name. "
    "Real websites almost always use a name, so be careful."
)


@dataclass(frozen=True)
class FeatureValue:
    """One feature the model received, with its value and a description."""

    name: str
    label: str
    value: float
    description: str


@dataclass(frozen=True)
class CheckResult:
    """Everything the web page needs to show for one checked URL.

    ``verdict`` is the yes/no decision at the threshold: ``"phishing"`` or
    ``"legitimate"``. ``band`` is the three-way risk band the CLI prints.
    ``contributions`` is ``None`` when the model cannot attribute its score
    to individual features, which is the case for the shipped forest.
    """

    input: str
    url: str
    verdict: str
    band: str
    score: float
    threshold: float
    model: str
    features: tuple[FeatureValue, ...]
    rule_flags: tuple[str, ...]
    contributions: tuple[tuple[str, float], ...] | None
    explanation: str


class CheckService:
    """Wrap a loaded ``Pipeline`` and turn a pasted URL into a ``CheckResult``."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    @classmethod
    def from_path(cls, path: Path = MODEL_PATH) -> CheckService:
        """Load a saved pipeline. Raises ``FileNotFoundError`` if it is missing."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run train.py first")
        return cls(Pipeline.load(path))

    @property
    def threshold(self) -> float:
        """The decision threshold saved with the model."""
        return self.pipeline.threshold

    @property
    def model_name(self) -> str:
        """Short name of the classifier, such as ``random_forest``."""
        return self.pipeline.model.name

    def check(self, raw: str) -> CheckResult:
        """Refang ``raw``, score it, and describe what the model saw."""
        text = raw.strip()
        url = refang(text)
        verdict = self.pipeline.score(url)
        matrix = self.pipeline.transform([url])
        names = self.pipeline.feature_names
        features = tuple(
            self._feature_value(name, float(value))
            for name, value in zip(names, matrix[0].tolist(), strict=True)
        )
        flags = self._rule_flags(url)
        contributions = self._contributions(matrix)
        is_phishing = verdict.probability >= self.threshold
        return CheckResult(
            input=text,
            url=url,
            verdict="phishing" if is_phishing else "legitimate",
            band=verdict.level.value,
            score=verdict.probability,
            threshold=self.threshold,
            model=self.model_name,
            features=features,
            rule_flags=flags,
            contributions=contributions,
            explanation=self._explain(verdict.probability, verdict.level, flags, contributions),
        )

    def _rule_flags(self, url: str) -> tuple[str, ...]:
        """Warnings from every extractor, plus a plain-English note for a raw IP host.

        The IP note is a web-only warning. The model already sees the same fact
        as its ``has_ip_host`` feature, so this does not change the score.
        """
        flags = [
            flag for extractor in self.pipeline.extractors for flag in extractor.rule_flags(url)
        ]
        if is_ip_address(hostname_of(url)):
            flags.append(IP_HOST_FLAG)
        return tuple(flags)

    @staticmethod
    def _feature_value(name: str, value: float) -> FeatureValue:
        """Attach a label and description, falling back to the raw name."""
        info = FEATURE_INFO.get(name, FeatureInfo(name, "A feature the model was trained on."))
        return FeatureValue(name, info.label, value, info.description)

    def _contributions(self, matrix: np.ndarray) -> tuple[tuple[str, float], ...] | None:
        """Per-feature contributions, largest first, or ``None`` if the model has none."""
        explained = self.pipeline.model.explain(matrix)
        if explained is None:
            return None
        pairs = zip(self.pipeline.feature_names, explained[0].tolist(), strict=True)
        return tuple(
            sorted(((n, float(v)) for n, v in pairs), key=lambda p: abs(p[1]), reverse=True)
        )

    def _explain(
        self,
        probability: float,
        level: RiskLevel,
        flags: tuple[str, ...],
        contributions: tuple[tuple[str, float], ...] | None,
    ) -> str:
        """A short paragraph saying what the score means and what it is based on."""
        side = "at or above" if probability >= self.threshold else "below"
        sentences = [
            f"The model gave this link a phishing score of {probability:.2f} "
            "on a scale from 0 to 1. "
            f"That is {side} the decision threshold of {self.threshold:.2f}, "
            f"so the result is {level.value}."
        ]
        if contributions is None:
            sentences.append(
                "The score comes from the hostname measurements listed below. "
                "This model is a forest of decision trees, which does not report how much "
                "each measurement contributed, so none of them is singled out as the cause."
            )
        else:
            pushing = [name for name, value in contributions if value > 0.3][:3]
            if pushing:
                sentences.append(
                    "The measurements that pushed the score up the most were "
                    + ", ".join(pushing)
                    + "."
                )
        if flags:
            sentences.append(
                "Separately, some simple rules fired. These are warnings for you to read "
                "and are not inputs to the model."
            )
        else:
            sentences.append("None of the simple warning rules fired.")
        sentences.append("The checker only reads the text of the link. It never opens the page.")
        return " ".join(sentences)
