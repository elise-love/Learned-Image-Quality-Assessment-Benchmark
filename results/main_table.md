| Method | Type | Trained on | SRCC | PLCC | seeds |
|---|---|---|---|---|---|
| NIQE | classical | none (natural-scene statistics) | 0.379 | 0.397 | - |
| BRISQUE | classical | LIVE (synthetic) | 0.231 | 0.248 | - |
| NIMA | learned | AVA (aesthetics) | 0.535 | 0.618 | - |
| NIMA | learned | KonIQ-10k | 0.880 | 0.911 | - |
| CLIP-IQA | zero-shot | none | 0.642 | 0.725 | - |
| MUSIQ | learned | KonIQ-10k | 0.870 | 0.902 | - |
| ResNet-50 (ours, backbone lr 1e-5) | fine-tuned | KonIQ-10k | 0.825 ± 0.007 | 0.853 ± 0.004 | 3 |
| ResNet-50 (ours, backbone lr 1e-4) | fine-tuned | KonIQ-10k | 0.892 | 0.911 | 1 |
| ViT-S/16 (ours) | fine-tuned | KonIQ-10k | 0.926 ± 0.001 | 0.937 ± 0.000 | 3 |
