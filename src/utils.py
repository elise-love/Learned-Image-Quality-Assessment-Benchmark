"""Shared helpers: seeding, logging, and IQA correlation metrics."""
import logging
import os
import random
import warnings

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


# ---------------------------------------------------------------- reproducibility
def set_seed(seed: int) -> None:
    """Fix every random source we use so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cudnn.benchmark=True is faster but non-deterministic; we accept small
        # GPU non-determinism and report mean +- std over seeds instead.
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def get_logger(name: str = "iqa", log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------- metrics
def logistic_4param(x, b1, b2, b3, b4):
    """Monotonic 4-parameter logistic used in IQA (VQEG convention).

    Maps raw model scores onto the MOS scale before computing PLCC, so PLCC
    measures prediction accuracy rather than penalizing a non-linear but
    monotonic relationship.
    """
    return (b1 - b2) / (1.0 + np.exp(-(x - b3) / (np.abs(b4) + 1e-8))) + b2


def fit_logistic(pred: np.ndarray, mos: np.ndarray) -> np.ndarray:
    """Return pred mapped through a fitted 4-param logistic.

    Falls back to a linear fit if the optimizer does not converge, so a bad
    fit never crashes an evaluation run (the fallback is logged as a warning).
    """
    pred = np.asarray(pred, dtype=np.float64)
    mos = np.asarray(mos, dtype=np.float64)
    p0 = [mos.max(), mos.min(), np.mean(pred), np.std(pred) + 1e-8]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params, _ = curve_fit(logistic_4param, pred, mos, p0=p0, maxfev=20000)
        return logistic_4param(pred, *params)
    except RuntimeError:
        warnings.warn("Logistic fit did not converge; using linear fit for PLCC.")
        a, b = np.polyfit(pred, mos, 1)
        return a * pred + b


def correlation_metrics(pred, mos, lower_better: bool = False) -> dict:
    """SRCC / PLCC / KRCC between predictions and MOS.

    Args:
        pred: model scores, one per image.
        mos:  ground-truth scores, higher = better quality.
        lower_better: True for metrics like NIQE/BRISQUE where a lower score
            means better quality. We flip their sign so every metric is
            reported in the same "higher correlation = better" direction.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    mos = np.asarray(mos, dtype=np.float64).ravel()
    ok = np.isfinite(pred) & np.isfinite(mos)
    pred, mos = pred[ok], mos[ok]
    if lower_better:
        pred = -pred

    srcc = stats.spearmanr(pred, mos)[0]
    krcc = stats.kendalltau(pred, mos)[0]
    plcc = stats.pearsonr(fit_logistic(pred, mos), mos)[0]
    return {"SRCC": float(srcc), "PLCC": float(plcc), "KRCC": float(krcc), "N": int(ok.sum())}
