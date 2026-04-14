"""
transforms.py
--------------
Görüntü ön işleme pipeline'ı.

ImageNet normalize değerleri kullanılır çünkü feature extractor
ImageNet ağırlıklarıyla başlatılmıştır.
"""

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transform(img_size: tuple[int, int]) -> T.Compose:
    """
    Standart inference transform pipeline'ı döner.

    Pipeline:
        1. Resize      → sabit boyuta getir
        2. ToTensor    → [0,255] uint8 → [0,1] float32 + (H,W,C) → (C,H,W)
        3. Normalize   → ImageNet mean/std ile normalize et

    Args:
        img_size: (H, W)

    Returns:
        torchvision transform compose
    """
    H, W = img_size
    return T.Compose([
        T.Resize((H, W)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(path: str, transform: T.Compose) -> torch.Tensor:
    """
    Görüntü dosyasını yükler ve transform uygular.

    Args:
        path      : görüntü dosya yolu
        transform : build_transform() ile oluşturulan pipeline

    Returns:
        tensor: (1, 3, H, W)  — batch dim eklenmiş
    """
    img = Image.open(path).convert("RGB")
    tensor = transform(img)       # (3, H, W)
    return tensor.unsqueeze(0)    # (1, 3, H, W)


def load_batch(paths: list[str], transform: T.Compose) -> torch.Tensor:
    """
    Birden fazla görüntüyü tek bir batch tensor'ına yükler.

    Args:
        paths     : görüntü dosya yolları listesi
        transform : build_transform() ile oluşturulan pipeline

    Returns:
        tensor: (B, 3, H, W)
    """
    tensors = [load_image(p, transform).squeeze(0) for p in paths]
    return torch.stack(tensors, dim=0)   # (B, 3, H, W)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Heatmap'i [0, 1] aralığına normalize eder.

    Args:
        heatmap: (H, W) veya (B, H, W)

    Returns:
        normalized: aynı shape, float32
    """
    h_min = heatmap.min()
    h_max = heatmap.max()
    return ((heatmap - h_min) / (h_max - h_min + 1e-8)).astype(np.float32)
