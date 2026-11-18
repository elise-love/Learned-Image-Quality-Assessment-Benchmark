"""Run pyiqa baselines and write per-image predictions + a summary table.

    python scripts/run_baselines.py --config configs/baselines.yaml
    python scripts/run_baselines.py --metrics niqe brisque --limit 50   # quick test

Outputs:
    results/baselines/<dataset>_<metric>.csv   per-image: name, pred, mos
    results/baselines/summary.csv              one row per (dataset, metric)
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import IQADataset  # noqa: E402
from src.utils import correlation_metrics, get_logger  # noqa: E402


@torch.no_grad()
def score_dataset(metric, loader, device):
    names, preds, mos = [], [], []
    for batch in tqdm(loader, leave=False):
        x = batch["img"].to(device)
        try:
            s = metric(x)
        except Exception:                      # some metrics only accept batch=1
            s = torch.cat([metric(x[i:i + 1]).reshape(1) for i in range(len(x))])
        preds += s.reshape(-1).float().cpu().tolist()
        names += list(batch["name"])
        mos += batch["mos"].tolist()
    return pd.DataFrame({"name": names, "pred": preds, "mos": mos})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baselines.yaml")
    ap.add_argument("--metrics", nargs="+", help="override metric list")
    ap.add_argument("--limit", type=int, help="only first N images (debug)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(cfg["out_dir"], exist_ok=True)
    log = get_logger(log_file=os.path.join(cfg["out_dir"], "run.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_names = args.metrics or list(cfg["metrics"])

    import pyiqa
    summary_path = os.path.join(cfg["out_dir"], "summary.csv")
    rows = pd.read_csv(summary_path).to_dict("records") if os.path.exists(summary_path) else []

    for ds_name, ds_cfg in cfg["datasets"].items():
        ds = IQADataset(ds_name, cfg["data_root"], phase=ds_cfg.get("phase"),
                        limit=args.limit)
        loader = DataLoader(ds, batch_size=ds_cfg.get("batch_size", 1),
                            num_workers=2, shuffle=False)
        log.info(f"{ds_name}: {len(ds)} images")

        for m in metric_names:
            metric = pyiqa.create_metric(m, device=device, as_loss=False)
            t0 = time.time()
            df = score_dataset(metric, loader, device)
            df.to_csv(os.path.join(cfg["out_dir"], f"{ds_name}_{m}.csv"), index=False)

            # pyiqa tells us each metric's direction; flip lower-is-better ones.
            res = correlation_metrics(df["pred"], df["mos"], metric.lower_better)
            log.info(f"{m:12s} SRCC {res['SRCC']:.4f}  PLCC {res['PLCC']:.4f}  "
                     f"({time.time() - t0:.0f}s, lower_better={metric.lower_better})")
            rows = [r for r in rows if not (r["dataset"] == ds_name and r["method"] == m)]
            rows.append({"dataset": ds_name, "method": m,
                         "trained_on": cfg["metrics"].get(m, {}).get("trained_on", ""),
                         "lower_better": metric.lower_better, **res,
                         "limit": args.limit or "full"})
            del metric
            torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    log.info(f"Summary written to {summary_path}")
    print(pd.DataFrame(rows)[["dataset", "method", "trained_on", "SRCC", "PLCC", "N"]]
          .to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
