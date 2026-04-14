"""
feature_extractor.py
---------------------
ResNet18'in erken katmanlarını kullanarak görüntüden
slope-representative feature map çıkarır.

Kullanılan katmanlar:
    stem   : conv1 + bn1 + relu + maxpool  → stride 4
    layer1 : BasicBlock x2                 → 64  kanal, stride 4
    layer2 : BasicBlock x2                 → 128 kanal, stride 8

Çıktı:
    layer1 ve layer2 aynı uzamsal boyuta getirilip concat edilir.
    Final shape: (B, 192, H/4, W/4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FeatureExtractor(nn.Module):
    """
    ResNet18 tabanlı shallow feature extractor.

    Args:
        pretrained (bool): ImageNet ağırlıklarını kullan (önerilir).
        freeze     (bool): Ağırlıkları dondur — sadece feature çıkarıcı olarak kullan.
    """

    # Çıktı kanal sayısı: layer1(64) + layer2(128) = 192
    OUT_CHANNELS = 192

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # ── Stem ────────────────────────────────────────────────────────
        # conv1: 7x7, stride 2  →  (B, 64, H/2, W/2)
        # maxpool: 3x3, stride 2 → (B, 64, H/4, W/4)
        self.stem = nn.Sequential(
            backbone.conv1,   # (B, 3,  H,   W  ) → (B, 64, H/2, W/2)
            backbone.bn1,
            backbone.relu,
            backbone.maxpool, # → (B, 64, H/4, W/4)
        )

        # ── Residual Blocks ─────────────────────────────────────────────
        self.layer1 = backbone.layer1  # (B, 64,  H/4, W/4)  — stride değişmez
        self.layer2 = backbone.layer2  # (B, 128, H/8, W/8)  — stride 2

        # layer3, layer4, avgpool, fc kullanılmıyor
        del backbone

        if freeze:
            self._freeze()

    # ── Private ─────────────────────────────────────────────────────────

    def _freeze(self):
        """Tüm parametreleri dondurur (gradient hesaplanmaz)."""
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  — normalize edilmiş RGB görüntü

        Returns:
            features: (B, 192, H/4, W/4)
        """
        x = self.stem(x)            # (B, 64,  H/4, W/4)
        f1 = self.layer1(x)         # (B, 64,  H/4, W/4)
        f2 = self.layer2(f1)        # (B, 128, H/8, W/8)

        # layer2 → layer1 boyutuna upsample (bilinear)
        f2_up = F.interpolate(
            f2,
            size=f1.shape[2:],      # (H/4, W/4)
            mode="bilinear",
            align_corners=False,
        )                           # (B, 128, H/4, W/4)

        # Kanallarda concat: 64 + 128 = 192
        features = torch.cat([f1, f2_up], dim=1)   # (B, 192, H/4, W/4)
        return features

    # ── Utility ─────────────────────────────────────────────────────────

    @property
    def out_channels(self) -> int:
        return self.OUT_CHANNELS

    def feature_shape(self, img_size: tuple[int, int]) -> tuple[int, int, int]:
        """Verilen img_size için çıktı shape'ini döner: (C, H/4, W/4)."""
        H, W = img_size
        return (self.OUT_CHANNELS, H // 4, W // 4)
