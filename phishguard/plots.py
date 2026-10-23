"""Charts for the README. Axes are labeled in plain English."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

PAPER = "#f5f3ee"
INK = "#1e2227"
CURVE_COLORS = {
    "logistic": "#2f5d50",
    "random_forest": "#b5473a",
    "gradient_boosting": "#3d5a80",
}
CURVE_LABELS = {
    "logistic": "logistic regression",
    "random_forest": "random forest",
    "gradient_boosting": "gradient boosting",
}


def save_confusion_matrix(matrix: list[list[int]], path: Path, title: str) -> None:
    """Save a 2 by 2 count matrix. Rows are actual, columns are predicted."""
    grid = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    ax.imshow(grid, cmap="Blues")
    ax.set_xticks([0, 1], ["predicted legitimate", "predicted phishing"])
    ax.set_yticks([0, 1], ["actual legitimate", "actual phishing"])
    ax.set_title(title, loc="left", fontsize=13, color=INK)
    darkest = float(grid.max()) if grid.size else 1.0
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            count = int(grid[row, col])
            color = "white" if grid[row, col] > darkest * 0.55 else INK
            ax.text(col, row, f"{count:,}", ha="center", va="center", color=color, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=PAPER)
    plt.close(fig)


def save_precision_recall(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    path: Path,
    winner: str,
    winner_threshold: float,
    winner_precision: float,
    winner_recall: float,
) -> None:
    """Save one precision-recall chart for every model on the test set.

    The star is the winner's measured precision and recall at the threshold
    chosen on the validation slice. It is not a point picked from this curve.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.6), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    for name, scores in probabilities.items():
        precision, recall, _thresholds = precision_recall_curve(labels, scores)
        ax.plot(
            recall,
            precision,
            color=CURVE_COLORS.get(name, INK),
            lw=2,
            label=CURVE_LABELS.get(name, name),
        )
    ax.scatter(
        [winner_recall],
        [winner_precision],
        s=80,
        zorder=5,
        color=CURVE_COLORS.get(winner, INK),
        edgecolor=INK,
        label=f"{CURVE_LABELS.get(winner, winner)} at {winner_threshold:.2f}",
    )
    ax.axhline(0.80, color="#8a8175", ls="--", lw=1, label="0.80 precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (phishing caught)")
    ax.set_ylabel("Precision (phishing alerts that are right)")
    ax.set_title("Precision-recall on the held-out test set", loc="left", fontsize=13, color=INK)
    ax.legend(frameon=False, loc="lower left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=PAPER)
    plt.close(fig)
