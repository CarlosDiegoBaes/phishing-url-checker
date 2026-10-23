"""Compare models on PhiUSIIL and write the charts and ``comparison.json``.

The saved checker (``model.joblib``) is the winning model, refit on the
full training split, at the threshold chosen on the validation slice.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from phishguard.compare import compare_models, comparison_payload, write_comparison
from phishguard.config import DEFAULT_MODEL, DEFAULT_THRESHOLD
from phishguard.features import LexicalExtractor
from phishguard.paths import DEFAULT_CSV, METRICS_PATH, MODEL_PATH, ROOT
from phishguard.pipeline import Pipeline
from phishguard.plots import save_confusion_matrix, save_precision_recall
from phishguard.train import load_urls, metrics_payload


def main() -> None:
    """Run the comparison and write the report artifacts."""
    if not DEFAULT_CSV.exists():
        sys.exit(
            "Dataset not found at data/PhiUSIIL_Phishing_URL_Dataset.csv. "
            "See the README for the download link."
        )
    urls, labels = load_urls(DEFAULT_CSV)
    extractor = LexicalExtractor()
    features = extractor.transform(urls)
    result = compare_models(features, np.asarray(labels))
    winner = result.winner_comparison()
    payload = comparison_payload(result)
    write_comparison(str(ROOT / "comparison.json"), result)

    pipeline = Pipeline(
        extractors=(extractor,),
        model=result.fitted_models[result.winner],
        threshold=winner.validation.threshold,
    )
    pipeline.save(MODEL_PATH)
    metrics_text = json.dumps(metrics_payload(pipeline, winner.test_at_chosen_threshold), indent=2)
    METRICS_PATH.write_text(metrics_text + "\n", encoding="utf-8")

    threshold = winner.validation.threshold
    tuned = winner.test_at_chosen_threshold
    save_confusion_matrix(
        tuned.confusion,
        ROOT / "confusion_matrix.png",
        f"{result.winner.replace('_', ' ')} at threshold {threshold:.2f}",
    )
    save_precision_recall(
        result.test_labels,
        result.test_probabilities,
        ROOT / "pr_curve.png",
        winner=result.winner,
        winner_threshold=threshold,
        winner_precision=tuned.precision,
        winner_recall=tuned.recall,
    )
    print(json.dumps(payload, indent=2))
    if (result.winner, threshold) != (DEFAULT_MODEL, DEFAULT_THRESHOLD):
        print(
            "config.py is out of date. Set "
            f"DEFAULT_MODEL = {result.winner!r} and DEFAULT_THRESHOLD = {threshold!r}."
        )


if __name__ == "__main__":
    main()
