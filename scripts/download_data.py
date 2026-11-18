"""Download datasets via pyiqa / Hugging Face.

    python scripts/download_data.py --datasets koniq10k            # phase 1
    python scripts/download_data.py --datasets spaq livec kadid10k # phase 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import SPECS, download, load_meta, sanity_check_direction  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["koniq10k"], choices=list(SPECS))
    ap.add_argument("--data-root", default="./datasets")
    args = ap.parse_args()

    for name in args.datasets:
        download(name, args.data_root)
        if name.startswith("koniq10k"):
            for phase in ["train", "val", "test"]:
                n = len(load_meta(name, args.data_root, phase))
                print(f"KonIQ-10k {phase}: {n} images")
            print("Expected official split: train 7058 / val 1000 / test 2015")
        else:
            df = load_meta(name, args.data_root)
            print(f"{name}: {len(df)} images, MOS range "
                  f"[{df.mos_raw.min():.2f}, {df.mos_raw.max():.2f}]")
            if name == "kadid10k":
                r = sanity_check_direction(df)
                print(f"KADID direction check: Spearman(level, mos) = {r:.3f} "
                      f"({'OK, higher = better' if r < 0 else 'WRONG DIRECTION, flip it'})")


if __name__ == "__main__":
    main()
