"""Evaluation: turn per-image prediction CSVs into SRCC / PLCC / KRCC tables.

Every experiment (baseline or trained model) writes a CSV with at least the
columns  name, pred, mos  (mos in [0,1], higher = better). This module is the
single place that computes the numbers, so all methods are scored identically.

Usage:
    python -m src.evaluate results/baselines/koniq10k_niqe.csv --lower-better
    python -m src.evaluate --summarize results/main_runs.csv   # mean +- std over seeds
"""
import argparse

import numpy as np
import pandas as pd

from src.utils import correlation_metrics


def evaluate_csv(path: str, lower_better: bool = False) -> dict:
    df = pd.read_csv(path)
    return correlation_metrics(df["pred"].values, df["mos"].values, lower_better)


def evaluate_by_group(df: pd.DataFrame, group_col: str, lower_better: bool = False,
                      min_size: int = 10) -> pd.DataFrame:
    """Per-group metrics, e.g. SRCC per KADID distortion type."""
    rows = []
    for g, sub in df.groupby(group_col):
        if len(sub) < min_size:
            continue
        m = correlation_metrics(sub["pred"].values, sub["mos"].values, lower_better)
        rows.append({group_col: g, **m})
    return pd.DataFrame(rows).sort_values("SRCC")


def summarize_seeds(df: pd.DataFrame, by=("method", "dataset"),
                    metrics=("SRCC", "PLCC")) -> pd.DataFrame:
    """Collapse rows that differ only by seed into 'mean +- std' strings."""
    agg = df.groupby(list(by))[list(metrics)].agg(["mean", "std", "count"])
    out = pd.DataFrame(index=agg.index)
    for m in metrics:
        mean, std, n = agg[(m, "mean")], agg[(m, "std")].fillna(0), agg[(m, "count")]
        out[m] = [f"{a:.3f} ± {b:.3f}" if c > 1 else f"{a:.3f}"
                  for a, b, c in zip(mean, std, n)]
    out["seeds"] = agg[(metrics[0], "count")]
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--lower-better", action="store_true")
    ap.add_argument("--summarize", help="CSV of per-run metrics with a 'seed' column")
    args = ap.parse_args()
    if args.summarize:
        print(summarize_seeds(pd.read_csv(args.summarize)).to_string(index=False))
    elif args.csv:
        res = evaluate_csv(args.csv, args.lower_better)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
