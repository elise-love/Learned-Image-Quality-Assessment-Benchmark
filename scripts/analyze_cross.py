"""Phase 3 analysis: cross-dataset table/figure, KADID per-distortion SRCC, error cases.

Reads
    results/cross/summary.csv + per-image CSVs            (our models, src.cross_eval)
    results/baselines/summary.csv                         (baselines on KonIQ test)
    results/baselines_cross/summary.csv + per-image CSVs  (baselines on SPAQ/LIVEC/KADID)
Writes
    results/cross/cross_table.{csv,md}
    results/kadid/per_type_srcc.csv
    results/figures/cross_dataset_srcc.png
    results/figures/kadid_per_type.png
    results/figures/errors_kadid.png, results/figures/errors_koniq.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import KADID_TYPES, SPECS  # noqa: E402
from src.utils import correlation_metrics, fit_logistic  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", "./datasets")
FIG = "results/figures"
DATASETS = ["koniq10k", "spaq", "livec", "kadid10k"]
DS_LABEL = {"koniq10k": "KonIQ-10k test\n(in-domain)", "spaq": "SPAQ\n(smartphone)",
            "livec": "LIVE Challenge", "kadid10k": "KADID-10k\n(synthetic)"}
OURS = {"vit_s16": "ViT-S/16 (ours)", "resnet50_lr1e-4": "ResNet-50 (ours)"}
BASE = {"musiq": "MUSIQ (KonIQ)", "nima-koniq": "NIMA (KonIQ)", "niqe": "NIQE", "brisque": "BRISQUE"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


# ---------------------------------------------------------------- loading
def load_scores() -> pd.DataFrame:
    """Long table: method, dataset, SRCC, PLCC (our models averaged over the seeds present)."""
    rows = []
    c = pd.read_csv("results/cross/summary.csv")
    for (m, d), g in c.groupby(["method", "dataset"]):
        rows.append({"method": m, "dataset": d, "SRCC": g.SRCC.mean(), "PLCC": g.PLCC.mean()})
    for path in ["results/baselines/summary.csv", "results/baselines_cross/summary.csv"]:
        if os.path.exists(path):
            b = pd.read_csv(path)
            b = b[b.dataset.isin(DATASETS) & b.method.isin(BASE)]
            rows += b[["method", "dataset", "SRCC", "PLCC"]].to_dict("records")
    return pd.DataFrame(rows).drop_duplicates(["method", "dataset"], keep="last")


def preds(method: str, dataset: str) -> tuple[pd.DataFrame, bool]:
    """Per-image predictions (first seed for our models) and the lower_better flag."""
    if method in OURS:
        runs = sorted(f for f in os.listdir("results/cross")
                      if f.startswith(f"{method}_seed") and f.endswith(f"_{dataset}.csv"))
        return pd.read_csv(os.path.join("results/cross", runs[0])), False
    lb = method in ("niqe", "brisque")
    return pd.read_csv(f"results/baselines_cross/{dataset}_{method}.csv"), lb


# ---------------------------------------------------------------- 1. cross-dataset
def cross_table(s: pd.DataFrame) -> pd.DataFrame:
    order = list(OURS) + list(BASE)
    t = s.pivot(index="method", columns="dataset", values="SRCC").reindex(order)[DATASETS]
    t.index = [{**OURS, **BASE}[m] for m in t.index]
    os.makedirs("results/cross", exist_ok=True)
    t.round(4).to_csv("results/cross/cross_table.csv")
    hdr = ["Method", "KonIQ-10k test", "SPAQ", "LIVE Challenge", "KADID-10k"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for name, r in t.iterrows():
        lines.append(f"| {name} | " + " | ".join("–" if pd.isna(v) else f"{v:.3f}" for v in r) + " |")
    open("results/cross/cross_table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return t


def cross_figure(t: pd.DataFrame):
    methods = [OURS["vit_s16"], OURS["resnet50_lr1e-4"], BASE["musiq"], BASE["nima-koniq"]]
    methods = [m for m in methods if m in t.index]
    x = np.arange(len(DATASETS))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, m in enumerate(methods):
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, t.loc[m, DATASETS].values, w * 0.9,
               color=COLORS[i], label=m, edgecolor="white", linewidth=1)
    ax.set_xticks(x, [DS_LABEL[d] for d in DATASETS], fontsize=9, color=INK2)
    ax.set_ylabel("SRCC", fontsize=9, color=INK2)
    ax.set_ylim(0, 1)
    ax.set_title("Trained on KonIQ-10k, tested without retraining", fontsize=10,
                 color=INK, loc="left")
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, ncol=len(methods), loc="upper center",
              bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(f"{FIG}/cross_dataset_srcc.png", dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 2. KADID per type
def kadid_per_type() -> pd.DataFrame:
    methods = [m for m in ["vit_s16", "resnet50_lr1e-4", "musiq"]]
    out = {}
    for m in methods:
        try:
            df, lb = preds(m, "kadid10k")
        except (FileNotFoundError, IndexError):
            continue
        df["dist_type"] = df["name"].str.split("_").str[1].astype(int)
        out[{**OURS, **BASE}[m]] = {k: correlation_metrics(g.pred, g.mos, lb)["SRCC"]
                                    for k, g in df.groupby("dist_type")}
    t = pd.DataFrame(out)
    t.index = [KADID_TYPES[i].replace("_", " ") for i in t.index]
    t = t.sort_values(t.columns[0])
    os.makedirs("results/kadid", exist_ok=True)
    t.round(4).to_csv("results/kadid/per_type_srcc.csv")

    fig, ax = plt.subplots(figsize=(7, 7.5))
    y = np.arange(len(t))
    for i, c in enumerate(t.columns):
        ax.scatter(t[c], y, s=40 if i == 0 else 26, color=COLORS[[0, 1, 2][i]], label=c,
                   zorder=3 - i, edgecolor="white", linewidth=1)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, t.index, fontsize=8, color=INK2)
    ax.set_xlabel("SRCC within distortion type (81 images x 5 levels)", fontsize=9, color=INK2)
    ax.set_title("KADID-10k: SRCC per distortion type (sorted by ViT-S/16)", fontsize=10,
                 color=INK, loc="left", pad=28)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(f"{FIG}/kadid_per_type.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(t.round(3).to_string())
    return t


# ---------------------------------------------------------------- 3. error cases
def error_grid(dataset: str, n: int = 8):
    df, _ = preds("vit_s16", dataset)
    df["pred_mapped"] = fit_logistic(df.pred.values, df.mos.values)
    df["err"] = df.pred_mapped - df.mos
    worst = df.reindex(df.err.abs().sort_values(ascending=False).index).head(n)
    img_dir = os.path.join(DATA_ROOT, SPECS[dataset].image_dir)
    fig, axes = plt.subplots(2, n // 2, figsize=(3.2 * n // 2, 6.0))
    for ax, (_, r) in zip(axes.ravel(), worst.iterrows()):
        ax.imshow(Image.open(os.path.join(img_dir, r["name"])).convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        label = ""
        if dataset == "kadid10k":
            p = r["name"].replace(".png", "").split("_")
            label = f"{KADID_TYPES[int(p[1])].replace('_', ' ')}, level {int(p[2])}\n"
        verdict = "over-rated" if r.err > 0 else "under-rated"
        ax.set_title(f"{label}MOS {r.mos:.2f} · predicted {r.pred_mapped:.2f} ({verdict})",
                     fontsize=8, color=INK)
    fig.suptitle(f"ViT-S/16: largest errors on {dataset}", fontsize=10, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/errors_{dataset}.png", dpi=150, facecolor="white")
    plt.close(fig)
    worst[["name", "mos", "pred_mapped", "err"]].to_csv(f"results/cross/worst_{dataset}.csv",
                                                       index=False)


def main():
    os.makedirs(FIG, exist_ok=True)
    s = load_scores()
    t = cross_table(s)
    cross_figure(t)
    kadid_per_type()
    for d in ["kadid10k", "koniq10k"]:
        try:
            error_grid(d)
        except FileNotFoundError as e:
            print(f"skip error grid for {d}: {e}")


if __name__ == "__main__":
    main()
