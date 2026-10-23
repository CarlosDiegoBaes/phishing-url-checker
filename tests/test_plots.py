"""Plot helpers write PNG files. The numbers in the files are the caller's."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phishguard.plots import save_confusion_matrix, save_precision_recall


def test_confusion_matrix_and_precision_recall_write_pngs(tmp_path: Path) -> None:
    matrix_path = tmp_path / "confusion_matrix.png"
    curve_path = tmp_path / "pr_curve.png"
    save_confusion_matrix(
        [[4, 1], [2, 3]],
        matrix_path,
        "toy model at threshold 0.40",
    )
    save_precision_recall(
        np.array([0, 0, 1, 1]),
        {
            "logistic": np.array([0.2, 0.4, 0.6, 0.8]),
            "random_forest": np.array([0.1, 0.3, 0.7, 0.9]),
            "gradient_boosting": np.array([0.2, 0.3, 0.8, 0.95]),
        },
        curve_path,
        winner="random_forest",
        winner_threshold=0.40,
        winner_precision=0.8,
        winner_recall=0.9,
    )
    assert matrix_path.read_bytes().startswith(b"\x89PNG")
    assert curve_path.read_bytes().startswith(b"\x89PNG")
