"""Zero-shot cross-dataset evaluation of models trained on KonIQ-10k.

The models are NOT retrained or recalibrated on the target datasets: we load the
best-on-val KonIQ-10k checkpoint and score every image of each target dataset.

    python -m src.cross_eval --runs vit_s16_seed0 resnet50_lr1e-4_seed0

Outputs:
    results/cross/<run>_<dataset>.csv   per-image: name, pred, mos (+ dist_type for KADID)
    results/cross/summary.csv           one row per (run, dataset)
"""
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets import IQADataset
from src.models import IQAModel
from src.train import predict
from src.utils import correlation_metrics, get_logger

# phase (split) and batch size per dataset. SPAQ images have different aspect
# ratios after resizing, so they are scored one at a time.
TARGETS = {
    "koniq10k": {"phase": "test", "batch_size": 32},   # in-domain reference
    "spaq": {"phase": None, "batch_size": 1},
    "livec": {"phase": None, "batch_size": 1},        # batch 1: robust to any image-size differences
    "kadid10k": {"phase": None, "batch_size": 32},
}


def load_model(ckpt_path: str, device) -> tuple[IQAModel, dict]:
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = st["cfg"]
    model = IQAModel(cfg["backbone"], pretrained=False,
                     freeze_backbone=cfg.get("freeze_backbone", False),
                     dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(st["model"])
    return model.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="e.g. vit_s16_seed0")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--datasets", nargs="+", default=list(TARGETS))
    ap.add_argument("--data-root", default="./datasets")
    ap.add_argument("--out", default="results/cross")
    ap.add_argument("--limit", type=int, help="debug: first N images per dataset")
    ap.add_argument("--force", action="store_true", help="re-score pairs already in the summary")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = get_logger("cross", os.path.join(args.out, "cross_eval.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_path = os.path.join(args.out, "summary.csv")
    rows = pd.read_csv(summary_path).to_dict("records") if os.path.exists(summary_path) else []

    for run in args.runs:
        model, cfg = load_model(os.path.join(args.ckpt_dir, run, "best.pth"), device)
        method, seed = run.rsplit("_seed", 1)
        for ds_name in args.datasets:
            if any(r["run"] == run and r["dataset"] == ds_name for r in rows) and not args.force:
                log.info(f"{run} {ds_name}: already in summary, skipping")
                continue
            t = TARGETS[ds_name]
            try:
                ds = IQADataset(ds_name, args.data_root, phase=t["phase"], limit=args.limit)
                dl = DataLoader(ds, batch_size=t["batch_size"], shuffle=False, num_workers=4,
                                pin_memory=device.type == "cuda")
                df = predict(model, dl, device)
            except Exception as e:                 # keep going with the other datasets
                log.info(f"{run} {ds_name}: FAILED with {type(e).__name__}: {e}")
                continue
            df.to_csv(os.path.join(args.out, f"{run}_{ds_name}.csv"), index=False)
            m = correlation_metrics(df["pred"], df["mos"])
            log.info(f"{run:24s} {ds_name:9s} SRCC {m['SRCC']:.4f}  PLCC {m['PLCC']:.4f}  N {m['N']}")
            rows = [r for r in rows if not (r["run"] == run and r["dataset"] == ds_name)]
            rows.append({"run": run, "method": method, "seed": int(seed), "dataset": ds_name, **m})
            pd.DataFrame(rows).to_csv(summary_path, index=False)   # save after every dataset
        del model
        torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    log.info(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
