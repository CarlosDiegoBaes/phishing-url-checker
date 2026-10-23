"""Locations of the dataset, the saved model, and the metrics file.

Paths are anchored to the project root (the parent of this package) so the
commands work no matter which directory they are launched from.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model.joblib"
METRICS_PATH = ROOT / "metrics.json"
DEFAULT_CSV = ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
