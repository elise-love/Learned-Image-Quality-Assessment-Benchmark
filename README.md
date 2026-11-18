# Learned Image Quality Assessment Benchmark

Do modern deep models fine-tuned on human quality ratings predict perceived image quality better than classical metrics, how well do they generalize across datasets, and on which distortion types do they fail?

> **Status: in progress.** Data pipeline, baselines, evaluation and the training pipeline are implemented; cross-dataset and distortion analyses follow. Tables below are filled in only with measured results.

## Motivation

In applied industry work on automatic photo quality scoring, we found that widely used no-reference metrics such as NIQE, BRISQUE and NIMA often disagreed with human judgments of real photos. This project measures that gap on public benchmarks. It fine-tunes CNN and ViT backbones on crowdsourced quality scores, compares them with classical and learned baselines under one evaluation protocol, and tests whether the learned models still work on data they were not trained on.

## Main results

KonIQ-10k official test split (2,015 images). SRCC is the primary metric. PLCC is computed after 4-parameter logistic fitting. Trained models: mean ± std over 3 seeds. All methods receive the same 512×384 input; baselines use pretrained weights from pyiqa without retraining.

| Method | Type | Trained on | SRCC | PLCC |
|---|---|---|---|---|
| NIQE | classical | none (natural-scene statistics) | 0.379 | 0.397 |
| BRISQUE | classical | LIVE (synthetic) | 0.231 | 0.248 |
| NIMA | learned | AVA (aesthetics) | 0.535 | 0.618 |
| NIMA | learned | KonIQ-10k | 0.880 | 0.911 |
| CLIP-IQA | zero-shot | none | 0.642 | 0.725 |
| MUSIQ | learned | KonIQ-10k | 0.870 | 0.902 |
| ResNet-50 (ours) | fine-tuned | KonIQ-10k | – | – |
| ViT-S/16 (ours) | fine-tuned | KonIQ-10k | – | – |

*(Figure `results/figures/scatter_koniq.png`, predicted score vs. MOS, will be added here.)*

## Key findings

To be written from measured results.

## Plan

| Experiment | Data | Question |
|---|---|---|
| Main benchmark | KonIQ-10k (train 7,058 / val 1,000 / test 2,015) | Do fine-tuned models beat classical and early learned metrics? |
| Cross-dataset | SPAQ (smartphone, ~11k), LIVE Challenge (1,162) | How much does accuracy drop without retraining? |
| Distortion analysis | KADID-10k (81 refs × 25 distortions × 5 levels) | Which distortion types are hardest? |
| Ablations (optional) | KonIQ-10k | Input resolution, frozen vs. fine-tuned backbone, loss choice |

**Protocol.** All scores are normalized to [0, 1] with higher = better. Lower-is-better metrics (NIQE, BRISQUE) are sign-flipped before computing correlations. Checkpoints are selected on the validation split, and the test split is evaluated once. Augmentation is limited to horizontal flips (and random crops where used), because color, brightness or blur changes would alter perceived quality and invalidate the labels.

**Models.** ImageNet-pretrained ResNet-50 and ViT-S/16 (timm), global pooling + `Linear(features, 1)`, L1 loss, AdamW (backbone lr 1e-5, head lr 1e-4), cosine schedule. KonIQ images are used at 512×384 without cropping.

## Reproduce

Designed for a free Colab or Kaggle GPU. The quickest route is [`notebooks/01_baselines.ipynb`](notebooks/01_baselines.ipynb).

```bash
pip install -r requirements.txt
python scripts/download_data.py --datasets koniq10k     # via pyiqa / Hugging Face
python scripts/run_baselines.py                          # writes results/baselines/
python -m src.train --config configs/overfit.yaml        # sanity check (64 images)
bash scripts/train_all.sh                                # 2 backbones x 3 seeds, tables, figure
```

Training entry point for Colab / Kaggle: [`notebooks/02_train.ipynb`](notebooks/02_train.ipynb). Every epoch saves a resumable checkpoint, and finished runs are skipped on rerun.

**Data.** Datasets are not stored in this repository. `download_data.py` fetches images and metadata (including the official KonIQ-10k split) from the [pyiqa dataset mirror](https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets). Please respect each dataset's original license: [KonIQ-10k](https://database.mmsp-kn.de/koniq-10k-database.html), [SPAQ](https://github.com/h4nwei/SPAQ), [LIVE Challenge](https://live.ece.utexas.edu/research/ChallengeDB/), [KADID-10k](https://database.mmsp-kn.de/kadid-10k-database.html).

## Repository layout

```
configs/    experiment configs (YAML)
src/        datasets.py, models.py, train.py, evaluate.py, utils.py
scripts/    download_data.py, run_baselines.py, train_all.sh, make_tables.py, plot_scatter.py
notebooks/  Colab / Kaggle entry points
results/    per-image predictions and summary tables (committed)
report/     technical report
```

## Limitations

- Pretrained MUSIQ and NIMA-KonIQ weights come from pyiqa and were trained on KonIQ-10k by their authors. They are strong in-domain references, not methods we trained ourselves.
- KADID-10k contains synthetic distortions, while KonIQ-10k contains authentic ones, so results there measure a domain shift as well as per-distortion difficulty.
- Free-tier GPUs limit model size and resolution; larger backbones are out of scope.

## Acknowledgements

Baselines and dataset mirrors are provided by [pyiqa (IQA-PyTorch)](https://github.com/chaofengc/IQA-PyTorch). Backbones come from [timm](https://github.com/huggingface/pytorch-image-models).
