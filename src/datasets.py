"""Dataset download + loading for KonIQ-10k, SPAQ, LIVE Challenge, KADID-10k.

Downloading uses pyiqa's helper (images + metadata are hosted on Hugging Face
under chaofengc/IQA-PyTorch-Datasets). Loading is our own small Dataset class
so we control normalization, splits and augmentation explicitly.

Every dataset returns images as float tensors in [0, 1] WITHOUT ImageNet
normalization: pyiqa baselines expect raw [0, 1] input, and our models apply
their own normalization inside forward(). That keeps one loader for both.
"""
import os
from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


@dataclass
class DatasetSpec:
    pyiqa_name: str      # name used by pyiqa.load_dataset (download only)
    image_dir: str       # relative to data_root
    meta_file: str       # relative to data_root
    mos_range: tuple     # raw score range in the metadata file
    lower_better: bool   # direction of the raw score
    split_col: str | None = None
    full_reference: bool = False   # KADID metadata has (ref, dist, score) columns
    skip_rows: int = 0             # LIVE Challenge: first 7 rows are its training images


# lower_better / mos_range follow pyiqa's default_dataset_configs.yml.
# KADID-10k: the column is called "dmos" but its values are higher = better
# (mean of degradation-category ratings, 1-5). sanity_check_direction() below
# verifies this from the data instead of trusting the column name.
SPECS = {
    "koniq10k": DatasetSpec("koniq10k", "koniq10k/512x384",
                            "meta_info/meta_info_KonIQ10kDataset.csv",
                            (1, 100), False, split_col="official_split"),
    # Same images and split at native resolution (resolution ablation for baselines)
    "koniq10k-1024": DatasetSpec("koniq10k-1024", "koniq10k/1024x768",
                                 "meta_info/meta_info_KonIQ10kDataset.csv",
                                 (1, 100), False, split_col="official_split"),
    "spaq": DatasetSpec("spaq", "SPAQ/resized512",
                        "meta_info/meta_info_SPAQDataset.csv", (1, 100), False),
    "livec": DatasetSpec("livec", "LIVEC",
                         "meta_info/meta_info_LIVEChallengeDataset.csv",
                         (1, 100), False, skip_rows=7),
    "kadid10k": DatasetSpec("kadid10k", "kadid10k/images",
                            "meta_info/meta_info_KADID10kDataset.csv",
                            (1, 5), False, full_reference=True),
}

# 25 KADID-10k distortion types, indexed by the 2nd field of "I01_05_03.png"
KADID_TYPES = {
    1: "gaussian_blur", 2: "lens_blur", 3: "motion_blur",
    4: "color_diffusion", 5: "color_shift", 6: "color_quantization",
    7: "color_saturation_1", 8: "color_saturation_2",
    9: "jpeg2000", 10: "jpeg",
    11: "white_noise", 12: "white_noise_color", 13: "impulse_noise",
    14: "multiplicative_noise", 15: "denoise",
    16: "brighten", 17: "darken", 18: "mean_shift",
    19: "jitter", 20: "non_eccentricity_patch", 21: "pixelate",
    22: "quantization", 23: "color_block",
    24: "high_sharpen", 25: "contrast_change",
}


# ---------------------------------------------------------------- download
def download(name: str, data_root: str = "./datasets") -> None:
    """Download images + metadata through pyiqa (idempotent)."""
    import pyiqa
    from pyiqa.data.dataset_api import load_dataset
    load_dataset(SPECS[name].pyiqa_name, data_root=data_root)
    if name == "spaq":
        resize_spaq(data_root)


def resize_spaq(data_root: str, short_side: int = 512) -> None:
    """SPAQ originals are ~4000 px; resize once so loading is not the bottleneck."""
    from tqdm import tqdm
    src = os.path.join(data_root, "SPAQ/TestImage")
    dst = os.path.join(data_root, SPECS["spaq"].image_dir)
    os.makedirs(dst, exist_ok=True)
    for f in tqdm(sorted(os.listdir(src)), desc="Resizing SPAQ"):
        out = os.path.join(dst, f)
        if os.path.exists(out):
            continue
        img = Image.open(os.path.join(src, f)).convert("RGB")
        w, h = img.size
        s = short_side / min(w, h)
        img.resize((round(w * s), round(h * s)), Image.BICUBIC).save(out, quality=95)


# ---------------------------------------------------------------- metadata
def load_meta(name: str, data_root: str, phase: str | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns: path, mos_raw, mos (in [0,1], higher=better),
    plus dist_type / dist_level for KADID."""
    spec = SPECS[name]
    meta = pd.read_csv(os.path.join(data_root, spec.meta_file))
    meta = meta.iloc[spec.skip_rows:].reset_index(drop=True)

    if spec.split_col is not None and phase is not None:
        meta = meta[meta[spec.split_col] == phase].reset_index(drop=True)

    if spec.full_reference:          # columns: ref_name, dist_name, score, ...
        names, scores = meta.iloc[:, 1], meta.iloc[:, 2]
    else:                            # columns: img_name, mos, ...
        names, scores = meta.iloc[:, 0], meta.iloc[:, 1]

    df = pd.DataFrame({
        "name": names.astype(str).values,
        "path": [os.path.join(data_root, spec.image_dir, n) for n in names],
        "mos_raw": scores.astype(float).values,
    })
    lo, hi = spec.mos_range
    df["mos"] = (df["mos_raw"] - lo) / (hi - lo)
    if spec.lower_better:
        df["mos"] = 1.0 - df["mos"]

    if name == "kadid10k":           # I01_05_03.png -> type 5, level 3
        parts = df["name"].str.replace(".png", "", regex=False).str.split("_")
        df["dist_type"] = parts.str[1].astype(int)
        df["dist_level"] = parts.str[2].astype(int)
        df["dist_name"] = df["dist_type"].map(KADID_TYPES)
    return df


def sanity_check_direction(df: pd.DataFrame) -> float:
    """KADID check: quality should drop as distortion level rises (1 -> 5).
    Returns Spearman(level, mos); it should be clearly negative."""
    from scipy.stats import spearmanr
    return float(spearmanr(df["dist_level"], df["mos"])[0])


# ---------------------------------------------------------------- torch Dataset
def build_transform(train: bool, crop_size: int | None = None,
                    resize: tuple | None = None) -> T.Compose:
    """Only geometry-preserving augmentation.

    We deliberately avoid color jitter, brightness, blur, etc.: those change the
    perceived quality of the image, so the MOS label would no longer be correct.
    """
    ops = []
    if resize is not None:                 # (H, W), e.g. for ViT fixed input
        ops.append(T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC,
                            antialias=True))
    if crop_size is not None:
        ops.append(T.RandomCrop(crop_size) if train else T.CenterCrop(crop_size))
    if train:
        ops.append(T.RandomHorizontalFlip())
    ops.append(T.ToTensor())               # -> [0, 1]
    return T.Compose(ops)


class IQADataset(Dataset):
    def __init__(self, name: str, data_root: str = "./datasets", phase: str | None = None,
                 train: bool = False, crop_size: int | None = None,
                 resize: tuple | None = None, limit: int | None = None):
        self.df = load_meta(name, data_root, phase)
        if limit is not None:              # tiny subsets for debugging / overfit tests
            self.df = self.df.iloc[:limit].reset_index(drop=True)
        self.transform = build_transform(train, crop_size, resize)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return {
            "img": self.transform(img),
            "mos": torch.tensor(row["mos"], dtype=torch.float32),
            "name": row["name"],
            "dist_type": int(row["dist_type"]) if "dist_type" in row else -1,
        }
