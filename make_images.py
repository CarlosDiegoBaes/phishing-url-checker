import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, ACC, RED = "#1e2227", "#d7dae0", "#98c379", "#e06c75"
lines = ["$ python checker.py <urls>", ""] + open("demo_output.txt").read().splitlines()
fig = plt.figure(figsize=(10, 0.32 * len(lines) + 0.6), facecolor=BG)
for i, ln in enumerate(lines):
    c = RED if "PHISHING" in ln else ACC if "SAFE" in ln or ln.startswith("$") else FG
    fig.text(0.03, 1 - (i + 1) / (len(lines) + 1), ln, family="DejaVu Sans Mono", fontsize=12, color=c)
fig.savefig("checker-demo.png", dpi=150, facecolor=BG); plt.close(fig)

m = json.load(open("metrics.json"))
names, w = zip(*reversed(m["weights"]))
fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#f5f3ee")
ax.set_facecolor("#f5f3ee")
ax.barh(names, w, color=["#b5473a" if x > 0 else "#2f5d50" for x in w])
ax.axvline(0, color="#333", lw=0.8)
ax.set_title("Logistic regression feature weights", fontsize=14, loc="left")
ax.set_xlabel("weight (standardized)  ·  positive = phishing")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("features.png", dpi=150, facecolor="#f5f3ee")
