"""Command-line checker. It never fetches the URL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phishguard.features import refang
from phishguard.paths import MODEL_PATH
from phishguard.pipeline import Pipeline
from phishguard.verdict import Verdict


def format_verdict(verdict: Verdict) -> str:
    """Render a verdict the same way the original checker printed it."""
    lines = [
        verdict.url,
        f"  verdict: {verdict.level.value}   p(phish) = {verdict.probability:.2f}",
    ]
    lines.extend(f"  - {reason}" for reason in verdict.reasons)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Arguments for scoring URLs and, optionally, printing feature weights."""
    parser = argparse.ArgumentParser(description="Score URLs for phishing risk.")
    parser.add_argument("urls", nargs="*", help="URLs to score (defanged hxxp/[.] is fine)")
    parser.add_argument("--weights", action="store_true", help="print feature weights")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="path to a saved pipeline (default: model.joblib in the project root)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Load a saved pipeline and print a verdict for each URL."""
    args = build_parser().parse_args(argv)
    if not args.model.exists():
        sys.exit("model.joblib not found; run train.py first")
    pipeline = Pipeline.load(args.model)
    if args.weights:
        _print_weights(pipeline)
    for url in args.urls:
        scored = pipeline.score(refang(url))
        shown = Verdict(url, scored.probability, scored.level, scored.reasons)
        print(format_verdict(shown))


def _print_weights(pipeline: Pipeline) -> None:
    weights = pipeline.feature_weights()
    if not weights:
        print("This model has no single weight per feature.")
        return
    for name, weight in weights:
        print(f"{name:>16}  {weight:+.2f}")


if __name__ == "__main__":
    main()
