"""Train on the PhiUSIIL Phishing URL Dataset (UCI) and save model.joblib."""
from __future__ import annotations

import json

import pandas as pd

from checker import PhishingClassifier

CSV = "data/PhiUSIIL_Phishing_URL_Dataset.csv"


def main() -> None:
    df = pd.read_csv(CSV, usecols=["URL", "label"])
    labels = (1 - df["label"]).tolist()  # dataset: 1 = legitimate; flip so 1 = phishing
    clf = PhishingClassifier()
    m = clf.train(df["URL"].tolist(), labels)
    clf.save()
    out = {"accuracy": m.accuracy, "precision": m.precision, "recall": m.recall,
           "confusion": m.confusion, "n_train": m.n_train, "n_test": m.n_test,
           "weights": clf.weights()}
    with open("metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
