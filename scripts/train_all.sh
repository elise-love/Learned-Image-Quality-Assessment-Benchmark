#!/usr/bin/env bash
# Train every main-experiment model with 3 seeds, then build tables and figures.
# Finished runs are skipped and interrupted runs resume, so re-running is safe.
#   bash scripts/train_all.sh                       # checkpoints in ./checkpoints
#   CKPT=/content/drive/MyDrive/learned-iqa/checkpoints bash scripts/train_all.sh
set -e
CKPT=${CKPT:-checkpoints}
SEEDS=${SEEDS:-"0 1 2"}

for cfg in configs/resnet50.yaml configs/vit_s16.yaml; do
  python -m src.train --config "$cfg" --seed $SEEDS --ckpt-dir "$CKPT"
done

python scripts/make_tables.py
python scripts/plot_scatter.py
