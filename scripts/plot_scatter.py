"""Predicted score vs. MOS on the KonIQ-10k test set, one panel per method.

Each panel shows every test image as a dot and the fitted 4-parameter
logistic curve (the same mapping used before computing PLCC).
Writes results/figures/scatter_koniq.png.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import correlation_metrics, logistic_4param  # noqa: E402

DOT = "#2a78d6"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

# (title, prediction csv, lower_better)
PANELS = [
    ("NIQE", "results/baselines/koniq10k_niqe.csv", True),
    ("NIMA (AVA)", "results/baselines/koniq10k_nima.csv", False),
    ("ResNet-50 (ours, seed 0)", "results/runs/resnet50_seed0/test_preds.csv", False),
    ("ViT-S/16 (ours, seed 0)", "results/runs/vit_s16_seed0/test_preds.csv", False),
]


def panel(ax, title, df, lower_better):
    x = -df.pred.values if lower_better else df.pred.values
    y = df.mos.values
    m = correlation_metrics(x, y)
    ax.scatter(x, y, s=6, c=DOT, alpha=0.35, linewidths=0)
    try:
        p0 = [y.max(), y.min(), x.mean(), x.std() + 1e-8]
        params, _ = curve_fit(logistic_4param, x, y, p0=p0, maxfev=20000)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, logistic_4param(xs, *params), color=INK, lw=2)
    except RuntimeError:
        pass
    ax.set_title(f"{title}\nSRCC {m['SRCC']:.3f} · PLCC {m['PLCC']:.3f}",
                 fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Predicted score" + (" (negated)" if lower_better else ""),
                  fontsize=9, color=INK2)
    ax.set_ylabel("MOS (normalized)", fontsize=9, color=INK2)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    panels = [(t, pd.read_csv(p), lb) for t, p, lb in PANELS if os.path.exists(p)]
    if not panels:
        print("No prediction files found yet.")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6),
                             squeeze=False)
    for ax, (t, df, lb) in zip(axes[0], panels):
        panel(ax, t, df, lb)
    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out = "results/figures/scatter_koniq.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
