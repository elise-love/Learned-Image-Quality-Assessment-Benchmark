"""IQA regression model: ImageNet-pretrained backbone + linear quality head.

    image [0,1] -> ImageNet normalization -> backbone -> global pool -> Linear -> score

The backbone comes from timm with num_classes=0, which removes the ImageNet
classifier and returns the pooled feature vector (2048-d for ResNet-50,
384-d for ViT-S/16).
"""
import timm
import torch
import torch.nn as nn

# Short names used in configs -> timm model names.
BACKBONES = {
    "resnet50": "resnet50.a1_in1k",                       # ImageNet-1k
    "vit_s16": "vit_small_patch16_384.augreg_in21k_ft_in1k",  # IN-21k pretrain, IN-1k fine-tune
}


class IQAModel(nn.Module):
    def __init__(self, backbone: str = "resnet50", pretrained: bool = True,
                 freeze_backbone: bool = False, dropout: float = 0.0):
        super().__init__()
        timm_name = BACKBONES.get(backbone, backbone)
        kwargs = {}
        if "vit" in timm_name:
            # Lets the ViT accept non-square inputs (KonIQ is 512x384) by
            # interpolating its positional embeddings, so we do not have to
            # crop or distort the image to a square.
            kwargs["dynamic_img_size"] = True
        self.backbone = timm.create_model(timm_name, pretrained=pretrained,
                                          num_classes=0, **kwargs)
        # Patch size for ViTs (None for CNNs, which accept any input size).
        pe = getattr(self.backbone, "patch_embed", None)
        self.patch = pe.patch_size[0] if pe is not None else None
        n_feat = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_feat, 1))

        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))

        if freeze_backbone:                     # ablation: linear probe only
            for p in self.backbone.parameters():
                p.requires_grad = False

    def init_head_bias(self, value: float) -> None:
        """Start the head at the mean training MOS, so early training learns
        the ranking of images instead of spending steps on a global offset."""
        nn.init.constant_(self.head[-1].bias, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch:                          # ViT: H and W must be multiples of 16
            h, w = x.shape[-2:]
            dh, dw = h % self.patch, w % self.patch
            if dh or dw:                        # center-crop at most 15 px per side pair
                x = x[..., dh // 2: h - (dh - dh // 2), dw // 2: w - (dw - dw // 2)]
        x = (x - self.mean) / self.std          # inputs arrive in [0, 1]
        return self.head(self.backbone(x)).squeeze(1)

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        """Separate learning rates: the pretrained backbone only needs small
        updates, while the randomly initialized head needs to learn quickly."""
        groups = [{"params": self.head.parameters(), "lr": lr_head}]
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        if bb:
            groups.append({"params": bb, "lr": lr_backbone})
        for g in groups:
            g["weight_decay"] = weight_decay
        return groups
