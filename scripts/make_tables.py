"""Combine baseline results and trained-model runs into one main results table.

Reads   results/baselines/summary.csv   (one row per baseline)
        results/main_runs.csv           (one row per trained model x seed)
Writes  results/main_table.csv and results/main_table.md (paste into README)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import summarize_seeds  # noqa: E402

DISPLAY = {  # pyiqa / config name -> (table name, type, trained on)
    "niqe": ("NIQE", "classical", "none (natural-scene statistics)"),
    "brisque": ("BRISQUE", "classical", "LIVE (synthetic)"),
    "nima": ("NIMA", "learned", "AVA (aesthetics)"),
    "nima-koniq": ("NIMA", "learned", "KonIQ-10k"),
    "clipiqa": ("CLIP-IQA", "zero-shot", "none"),
    "musiq": ("MUSIQ", "learned", "KonIQ-10k"),
    "resnet50": ("ResNet-50 (ours, backbone lr 1e-5)", "fine-tuned", "KonIQ-10k"),
    "resnet50_lr1e-4": ("ResNet-50 (ours, backbone lr 1e-4)", "fine-tuned", "KonIQ-10k"),
    "vit_s16": ("ViT-S/16 (ours)", "fine-tuned", "KonIQ-10k"),
}


def main(dataset="koniq10k"):
    parts = []
    b_path = "results/baselines/summary.csv"
    if os.path.exists(b_path):
        b = pd.read_csv(b_path)
        b = b[(b.dataset == dataset) & (b["limit"].astype(str) == "full")]
        b = b.assign(SRCC=b.SRCC.map("{:.3f}".format), PLCC=b.PLCC.map("{:.3f}".format),
                     seeds="-")
        parts.append(b[["method", "SRCC", "PLCC", "seeds"]])
    r_path = "results/main_runs.csv"
    if os.path.exists(r_path):
        r = pd.read_csv(r_path)
        parts.append(summarize_seeds(r[r.dataset == dataset])[["method", "SRCC", "PLCC", "seeds"]])
    if not parts:
        print("No results yet.")
        return

    t = pd.concat(parts, ignore_index=True)
    t["order"] = t.method.map({k: i for i, k in enumerate(DISPLAY)}).fillna(99)
    t = t.sort_values("order")
    t["Method"] = t.method.map(lambda m: DISPLAY.get(m, (m,))[0])
    t["Type"] = t.method.map(lambda m: DISPLAY.get(m, ("", ""))[1])
    t["Trained on"] = t.method.map(lambda m: DISPLAY.get(m, ("", "", ""))[2])
    out = t[["Method", "Type", "Trained on", "SRCC", "PLCC", "seeds"]]
    out.to_csv("results/main_table.csv", index=False)

    lines = ["| " + " | ".join(out.columns) + " |", "|" + "---|" * len(out.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in out.values]
    open("results/main_table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
