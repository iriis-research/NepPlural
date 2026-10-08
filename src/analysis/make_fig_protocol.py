"""
Figure 1 for the camera-ready: agreement vs validity across annotation protocols.
Reads results/protocol_audit.json (written by protocol_audit.py), so the figure
always matches the reported numbers.

Usage (repo root):  python src/analysis/make_fig_protocol.py
Writes:             paper/figures/protocol_audit.pdf (+ .png preview)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("results/protocol_audit.json", encoding="utf-8"))
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

groups = ["Human", "Batched", "Forced", "Abstain"]
kappa = [np.mean(d["kappa"][k]) for k in
         ["humans | all", "batched-chat | all", "forced | all", "abstain | all"]]
f1 = [np.mean(d["loo"][k]) for k in
      ["human (held-out)", "batched-chat | majority", "forced | majority", "abstain | majority"]]
colors = ["#4C4C4C", "#C0504D", "#E3A33B", "#2E75B6"]

plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.75), sharey=True)
for ax, vals, title in [(axes[0], kappa, "(a) Agreement (Fleiss' $\\kappa$)"),
                        (axes[1], f1, "(b) Validity (LOO macro-F1)")]:
    bars = ax.bar(range(4), vals, color=colors, width=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6)
    ax.axhline(vals[0], color=colors[0], lw=0.6, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels(groups, fontsize=6)
    ax.set_title(title, fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, width=0.5)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(OUT / "protocol_audit.pdf", bbox_inches="tight")
fig.savefig(OUT / "protocol_audit.png", dpi=200, bbox_inches="tight")
print("kappa:", [round(x, 2) for x in kappa], " LOO F1:", [round(x, 2) for x in f1])
print(f"Saved {OUT/'protocol_audit.pdf'}")
