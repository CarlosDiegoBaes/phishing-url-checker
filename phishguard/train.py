"""Train the checker on PhiUSIIL and write ``model.joblib``.

The dataset stores ``1`` for a legitimate URL. This module flips that so
``1`` means phishing, which is the class precision and recall describe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from phishguard.config import DEFAULT_MODEL, DEFAULT_THRESHOLD
from phishguard.evaluate import metrics_from_probabilities, split_features
from phishguard.features import LexicalExtractor
from phishguard.models import LogisticModel, Model
from phishguard.paths import DEFAULT_CSV, METRICS_PATH, MODEL_PATH
from phishguard.pipeline import Pipeline
from phishguard.registry import default_registry
from phishguard.verdict import Metrics


def load_urls(path: Path) -> tuple[list[str], list[int]]:
    """Read URL strings and phishing labels from a PhiUSIIL-style CSV.

    The file's ``label`` column uses ``1`` for legitimate. The returned
    labels use ``1`` for phishing.
    """
    frame = pd.read_csv(path, usecols=["URL", "label"])
    urls = frame["URL"].tolist()
    labels = (1 - frame["label"]).astype(int).tolist()
    return urls, labels


def fit_holdout(
    urls: list[str],
    labels: list[int],
    model: Model,
    threshold: float = 0.5,
) -> tuple[Pipeline, Metrics]:
    """Fit ``model`` on the stratified training split and score the holdout.

    The saved model sees only the training rows. Seed 42 and a 20% test
    set match the original checker.
    """
    extractor = LexicalExtractor()
    features = extractor.transform(urls)
    label_array = np.asarray(labels)
    x_train, x_test, y_train, y_test = split_features(features, label_array)
    model.fit(x_train, y_train)
    metrics = metrics_from_probabilities(
        y_test,
        model.predict_proba(x_test),
        threshold,
        n_train=len(y_train),
        model_name=model.name,
    )
    pipeline = Pipeline(extractors=(extractor,), model=model, threshold=threshold)
    return pipeline, metrics


def train(urls: list[str], labels: list[int], threshold: float = 0.5) -> tuple[Pipeline, Metrics]:
    """Fit the baseline logistic regression and score it at ``threshold``.

    The default threshold of 0.5 is the published baseline.
    """
    return fit_holdout(urls, labels, LogisticModel(), threshold)


def metrics_payload(pipeline: Pipeline, metrics: Metrics) -> dict[str, object]:
    """Metrics plus the sorted feature weights, ready to write as JSON."""
    payload = metrics.as_dict()
    payload["weights"] = [[name, weight] for name, weight in pipeline.feature_weights()]
    return payload


def main() -> None:
    """Train on the CSV next to the repo and write the model and metrics files."""
    if not DEFAULT_CSV.exists():
        sys.exit(
            "Dataset not found at data/PhiUSIIL_Phishing_URL_Dataset.csv. "
            "See the README for the download link."
        )
    urls, labels = load_urls(DEFAULT_CSV)
    model = default_registry().create(DEFAULT_MODEL)
    pipeline, metrics = fit_holdout(urls, labels, model, DEFAULT_THRESHOLD)
    pipeline.save(MODEL_PATH)
    payload = metrics_payload(pipeline, metrics)
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
