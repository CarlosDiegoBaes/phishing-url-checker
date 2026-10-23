"""Redraw the README images from demo_output.txt and metrics.json.

Run this after ``python train.py`` and after refreshing ``demo_output.txt``.
The feature-weight chart reads the weights the trainer just wrote.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, ACC, RED = "#1e2227", "#d7dae0", "#98c379", "#e06c75"
with open("demo_output.txt", encoding="utf-8") as demo_file:
    demo_lines = demo_file.read().splitlines()
lines = ["$ python checker.py <urls>", ""] + demo_lines
fig = plt.figure(figsize=(10, 0.32 * len(lines) + 0.6), facecolor=BG)
for i, ln in enumerate(lines):
    color = RED if "PHISHING" in ln else ACC if "SAFE" in ln or ln.startswith("$") else FG
    fig.text(
        0.03,
        1 - (i + 1) / (len(lines) + 1),
        ln,
        family="DejaVu Sans Mono",
        fontsize=12,
        color=color,
    )
fig.savefig("checker-demo.png", dpi=150, facecolor=BG)
plt.close(fig)

with open("metrics.json", encoding="utf-8") as metrics_file:
    metrics = json.load(metrics_file)
pairs = list(reversed(metrics["weights"]))
names = [name for name, _weight in pairs]
weights = [weight for _name, weight in pairs]
fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#f5f3ee")
ax.set_facecolor("#f5f3ee")
ax.barh(names, weights, color=["#b5473a" if value > 0 else "#2f5d50" for value in weights])
ax.axvline(0, color="#333", lw=0.8)
ax.set_title("Logistic regression feature weights", fontsize=14, loc="left")
ax.set_xlabel("weight (standardized)  ·  positive = phishing")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig("features.png", dpi=150, facecolor="#f5f3ee")
