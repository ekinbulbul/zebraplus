"""
inspector.py
-------------
Paylaşılan tek bir backbone (FeatureExtractor) üzerinde
10 bağımsız GaussianModel çalıştırır.

Bellek düzeni:
    - FeatureExtractor : 1×  (ResNet18 early layers, ~11 MB)
    - GaussianModel    : 10× (her biri kendi .npz ağırlığıyla)

Yönlendirme:
    crop_id % 10  →  gaussian[i]

Ağırlık klasörü beklentisi:
    weights/
        model_0/gaussian_model.npz
        model_1/gaussian_model.npz
        ...
        model_9/gaussian_model.npz

Kullanım:
    inspector = CropInspector(weights_dir="weights/", device="cpu")
    result = inspector.inspect(crop_id=7, image_path="crop_007.png")
    # {"crop_id": 7, "model_idx": 7, "score": 2.34, "label": "NORMAL", "heatmap": ...}
"""

from __future__ import annotations

import torch
import numpy as np
import cv2
from pathlib import Path

from model.feature_extractor import FeatureExtractor
from model.gaussian_model    import GaussianModel
from core.transforms         import build_transform, load_batch, normalize_heatmap

NUM_MODELS = 10


class CropInspector:
    """
    Tek backbone + 10 bağımsız GaussianModel.

    Args:
        weights_dir : weights/ klasörü; altında model_0/ ... model_9/ beklenir
        img_size    : (H, W) görüntü boyutu
        device      : "cuda" | "cpu" | None (otomatik)
        pretrained  : backbone ImageNet ağırlıkları (önerilir)
    """

    def __init__(
        self,
        weights_dir: str,
        img_size: tuple[int, int] = (256, 256),
        device: str | None = None,
        pretrained: bool = True,
        half: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device    = torch.device(device)
        self.img_size  = img_size
        self.half      = half
        self.transform = build_transform(img_size)

        # ── Paylaşılan backbone (1×) ─────────────────────────────────
        print(f"[CropInspector] Backbone yükleniyor → device={self.device}")
        self.backbone = FeatureExtractor(pretrained=pretrained, freeze=True).to(self.device)
        self.backbone.eval()

        # ── 10 bağımsız GaussianModel ────────────────────────────────
        print(f"[CropInspector] {NUM_MODELS} GaussianModel yükleniyor...")
        self.gaussians: list[GaussianModel] = []
        dtype_str = "float16" if half else "float32"
        for i in range(NUM_MODELS):
            g = GaussianModel(device=self.device)
            model_path = str(Path(weights_dir) / f"model_{i}" / "gaussian_model")
            g.load(model_path, half=half)
            self.gaussians.append(g)
            print(f"  gaussian[{i}] ← {model_path}.npz  [{dtype_str}]")

        print("[CropInspector] Hazır.")

    # ── İç yardımcılar ───────────────────────────────────────────────

    def _route(self, crop_id: int) -> int:
        """crop_id % 10 → gaussian index."""
        return crop_id % 10

    @torch.no_grad()
    def _extract(self, image_path: str) -> torch.Tensor:
        """Görüntüyü yükler, backbone'dan feature map çıkarır."""
        tensor = load_batch([image_path], self.transform).to(self.device)
        return self.backbone(tensor)           # (1, 192, H/4, W/4)

    def _score(self, features: torch.Tensor, model_idx: int) -> tuple[np.ndarray, float]:
        """Feature map'i ilgili Gaussian'a gönderir, heatmap + skor döner."""
        dist_map = self.gaussians[model_idx].mahalanobis_map(features)[0]  # (h, w)
        dist_np  = dist_map.float().cpu().numpy()
        H, W     = self.img_size
        resized  = cv2.resize(dist_np, (W, H), interpolation=cv2.INTER_LINEAR)
        score    = float(resized.mean())
        heatmap  = normalize_heatmap(resized)
        return heatmap, score

    # ── Dışa açık API ────────────────────────────────────────────────

    def inspect(
        self,
        crop_id: int,
        image_path: str,
        threshold: float | None = None,
    ) -> dict:
        """
        Tek bir crop'u analiz eder.

        Returns:
            {
                "crop_id"   : int,
                "model_idx" : int,
                "score"     : float,
                "label"     : "DEFECT" | "NORMAL" | "UNKNOWN",
                "heatmap"   : np.ndarray  (H, W) float32 [0,1]
            }
        """
        model_idx      = self._route(crop_id)
        features       = self._extract(image_path)
        heatmap, score = self._score(features, model_idx)

        label = "UNKNOWN"
        if threshold is not None:
            label = "DEFECT" if score >= threshold else "NORMAL"

        return {
            "crop_id"   : crop_id,
            "model_idx" : model_idx,
            "score"     : round(score, 4),
            "label"     : label,
            "heatmap"   : heatmap,
        }

    def inspect_batch(
        self,
        crops: list[tuple[int, str]],
        threshold: float | None = None,
    ) -> list[dict]:
        """
        [(crop_id, image_path), ...] listesini sırayla işler.
        Backbone her crop için bir kez çalışır.
        """
        results = []
        for crop_id, image_path in crops:
            results.append(self.inspect(crop_id, image_path, threshold))
        return results
