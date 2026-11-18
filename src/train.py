"""Fine-tune an IQA model on KonIQ-10k.

    python -m src.train --config configs/resnet50.yaml --seed 0
    python -m src.train --config configs/overfit.yaml          # sanity check
    python -m src.train --config configs/resnet50.yaml --seed 0 \
        --ckpt-dir /content/drive/MyDrive/learned-iqa/checkpoints  # Colab

Protocol:
  * train on the official train split, pick the best epoch by val SRCC
  * evaluate the best checkpoint on the test split exactly once, at the end
  * every epoch saves last.pth, so an interrupted run resumes where it stopped
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.datasets import IQADataset
from src.models import IQAModel
from src.utils import correlation_metrics, get_logger, set_seed


# ---------------------------------------------------------------- config
def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(open(path))
    if "base" in cfg:                       # simple inheritance: base: configs/x.yaml
        base = yaml.safe_load(open(cfg.pop("base")))
        base.update(cfg)
        cfg = base
    for kv in overrides:                    # --set epochs=2 batch_size=8
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


# ---------------------------------------------------------------- loss
def plcc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson correlation within the batch. Rewards getting the *relative*
    ordering/linearity right, which is what SRCC/PLCC measure."""
    p = pred - pred.mean()
    t = target - target.mean()
    return 1 - (p * t).sum() / (p.norm() * t.norm() + 1e-8)


def compute_loss(pred, target, cfg):
    kind = cfg.get("loss", "l1")
    base = F.mse_loss(pred, target) if kind.startswith("mse") else F.l1_loss(pred, target)
    if "plcc" in kind:
        base = base + cfg.get("plcc_weight", 1.0) * plcc_loss(pred, target)
    return base


# ---------------------------------------------------------------- data
def make_loaders(cfg: dict):
    common = dict(name=cfg["dataset"], data_root=cfg["data_root"],
                  crop_size=cfg.get("crop_size"), resize=cfg.get("resize"))
    n_overfit = cfg.get("overfit")
    if n_overfit:
        # Train and "validate" on the same few images with no augmentation:
        # a correct pipeline must drive the loss close to zero.
        train_ds = IQADataset(phase="train", train=False, limit=n_overfit, **common)
        val_ds = test_ds = train_ds
    else:
        train_ds = IQADataset(phase="train", train=True, **common)
        val_ds = IQADataset(phase="val", train=False, **common)
        test_ds = IQADataset(phase="test", train=False, **common)

    kw = dict(num_workers=cfg.get("num_workers", 4), pin_memory=torch.cuda.is_available())
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          drop_last=not n_overfit, persistent_workers=kw["num_workers"] > 0,
                          **kw)
    ev = cfg.get("eval_batch_size", cfg["batch_size"] * 2)
    val_dl = DataLoader(val_ds, batch_size=ev, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, batch_size=ev, shuffle=False, **kw)
    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------- eval
@torch.no_grad()
def predict(model, loader, device, amp: bool = True) -> pd.DataFrame:
    """Per-image predictions. Reused for cross-dataset tests in phase 3."""
    model.eval()
    names, preds, mos, dtypes = [], [], [], []
    for b in loader:
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            p = model(b["img"].to(device, non_blocking=True))
        preds += p.float().cpu().tolist()
        mos += b["mos"].tolist()
        names += list(b["name"])
        dtypes += b["dist_type"].tolist()
    df = pd.DataFrame({"name": names, "pred": preds, "mos": mos})
    if any(d >= 0 for d in dtypes):
        df["dist_type"] = dtypes
    return df


# ---------------------------------------------------------------- main loop
def train(cfg: dict, seed: int, ckpt_root: str, force: bool = False) -> dict | None:
    run_name = f"{cfg['exp_name']}_seed{seed}"
    ckpt_dir = os.path.join(ckpt_root, run_name)
    out_dir = os.path.join(cfg.get("out_dir", "results/runs"), run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    log = get_logger(run_name, os.path.join(out_dir, "train.log"))

    done_flag = os.path.join(out_dir, "test_metrics.json")
    if os.path.exists(done_flag) and not force:
        log.info(f"{run_name} already finished; skipping (use --force to rerun).")
        return json.load(open(done_flag))

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.get("amp", True) and device.type == "cuda"
    train_dl, val_dl, test_dl = make_loaders(cfg)
    log.info(f"train {len(train_dl.dataset)} | val {len(val_dl.dataset)} | "
             f"test {len(test_dl.dataset)} | device {device}")

    model = IQAModel(cfg["backbone"], pretrained=cfg.get("pretrained", True),
                     freeze_backbone=cfg.get("freeze_backbone", False),
                     dropout=cfg.get("dropout", 0.0)).to(device)
    if cfg.get("init_bias_to_mean", True):
        model.init_head_bias(float(train_dl.dataset.df["mos"].mean()))
    opt = torch.optim.AdamW(model.param_groups(cfg["lr_backbone"], cfg["lr_head"],
                                               cfg.get("weight_decay", 0.01)))

    # Linear warm-up then cosine decay, updated every step.
    epochs = cfg["epochs"]
    total = epochs * len(train_dl)
    warm = int(cfg.get("warmup_epochs", 1) * len(train_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best_srcc, best_epoch, history = 0, -1.0, -1, []
    last_path = os.path.join(ckpt_dir, "last.pth")
    best_path = os.path.join(ckpt_dir, "best.pth")
    if os.path.exists(last_path):           # resume an interrupted run
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start_epoch, best_srcc, best_epoch = st["epoch"] + 1, st["best_srcc"], st["best_epoch"]
        history = st["history"]
        log.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0, losses = time.time(), []
        for b in train_dl:
            x = b["img"].to(device, non_blocking=True)
            y = b["mos"].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                pred = model(x)
            loss = compute_loss(pred.float(), y, cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        val = correlation_metrics(*predict(model, val_dl, device, amp)[["pred", "mos"]].T.values)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_SRCC": val["SRCC"], "val_PLCC": val["PLCC"],
               "lr_head": opt.param_groups[0]["lr"], "sec": time.time() - t0}
        history.append(rec)
        log.info(f"ep {epoch:2d} | loss {rec['train_loss']:.4f} | val SRCC {val['SRCC']:.4f} "
                 f"PLCC {val['PLCC']:.4f} | {rec['sec']:.0f}s")

        if val["SRCC"] > best_srcc:
            best_srcc, best_epoch = val["SRCC"], epoch
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_srcc": best_srcc, "best_epoch": best_epoch,
                    "history": history}, last_path)
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # ---- final, single test evaluation with the best-on-val checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    preds = predict(model, test_dl, device, amp)
    preds.to_csv(os.path.join(out_dir, "test_preds.csv"), index=False)
    res = {"method": cfg["exp_name"], "dataset": cfg["dataset"], "seed": seed,
           **correlation_metrics(preds["pred"], preds["mos"]),
           "best_epoch": best_epoch, "val_SRCC": best_srcc}
    log.info(f"TEST | SRCC {res['SRCC']:.4f} | PLCC {res['PLCC']:.4f} | best epoch {best_epoch}")
    json.dump(res, open(done_flag, "w"), indent=2)

    if not cfg.get("overfit"):              # only real runs go in the results table
        table = cfg.get("results_table", "results/main_runs.csv")
        rows = pd.read_csv(table) if os.path.exists(table) else pd.DataFrame()
        if len(rows):
            rows = rows[~((rows.method == res["method"]) & (rows.seed == seed))]
        pd.concat([rows, pd.DataFrame([res])]).to_csv(table, index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--force", action="store_true", help="rerun finished runs")
    ap.add_argument("--set", nargs="*", default=[], help="override config: key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    for s in args.seed:
        train(cfg, s, args.ckpt_dir, args.force)


if __name__ == "__main__":
    main()
