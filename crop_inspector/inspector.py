"""
inspector.py
-------------
Paylaşılan tek backbone (FeatureExtractor) üzerinde
200 bağımsız GaussianModel çalıştırır.

Yönlendirme : crop_id % NUM_MODELS → gaussian[i]

Ağırlık klasörü:
    weights/
        model_0/gaussian_model.npz
        model_0/meta.txt
        ...
        model_9/gaussian_model.npz
        model_9/meta.txt
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path

from model.feature_extractor import FeatureExtractor
from model.gaussian_model    import GaussianModel
from core.transforms         import build_transform, load_batch, normalize_heatmap

NUM_MODELS = 10


def _load_meta(model_dir: Path) -> dict:
    """meta.txt varsa okur, yoksa varsayılan döner."""
    meta_path = model_dir / "meta.txt"
    meta = {"pool": 1, "img_size": (1200, 1200)}
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k == "pool":
                meta["pool"] = int(v)
            elif k == "img_size":
                h, w = v.split(",")
                meta["img_size"] = (int(h), int(w))
    return meta


class CropInspector:
    """
    Tek backbone + NUM_MODELS bağımsız GaussianModel.

    Args:
        weights_dir : weights/ klasörü; altında model_0/ ... model_9/ beklenir
        img_size    : (H, W) — inference görüntü boyutu
        device      : "cuda" | "cpu" | None (otomatik)
        pretrained  : backbone ImageNet ağırlıkları
    """

    def __init__(
        self,
        weights_dir: str,
        img_size: tuple[int, int] = (1200, 1200),
        device: str | None = None,
        pretrained: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device    = torch.device(device)
        self.img_size  = img_size
        self.transform = build_transform(img_size)

        # ── Paylaşılan backbone (1×) ─────────────────────────────────
        print(f"[CropInspector] Backbone yükleniyor → device={self.device}")
        self.backbone = FeatureExtractor(pretrained=pretrained, freeze=True).to(self.device)
        self.backbone.eval()

        # ── NUM_MODELS bağımsız GaussianModel (CPU RAM'de bekler) ────
        print(f"[CropInspector] {NUM_MODELS} GaussianModel yükleniyor (CPU)...")
        self.gaussians: list[GaussianModel] = []
        self.pools: list[int] = []

        for i in range(NUM_MODELS):
            model_dir  = Path(weights_dir) / f"model_{i}"
            meta       = _load_meta(model_dir)
            pool       = meta["pool"]

            # Gaussian CPU'da yüklenir — Mahalanobis anında GPU'ya taşınır
            g = GaussianModel(device="cpu")
            g.load(str(model_dir / "gaussian_model"))
            self.gaussians.append(g)
            self.pools.append(pool)
            print(f"  gaussian[{i}] pool=x{pool} ← {model_dir}")

        print("[CropInspector] Hazır.")

    # ── İç yardımcılar ───────────────────────────────────────────────

    def _route(self, crop_id: int) -> int:
        return crop_id % NUM_MODELS

    @torch.no_grad()
    def _extract(self, image_path: str) -> torch.Tensor:
        """Backbone forward — sonuç GPU'da kalır."""
        tensor = load_batch([image_path], self.transform).to(self.device)
        return self.backbone(tensor)      # (1, 192, H/4, W/4)

    def _score(self, features: torch.Tensor, model_idx: int) -> tuple[np.ndarray, float]:
        """
        Precision'ı CPU'dan GPU'ya taşı → Mahalanobis hesapla → CPU'ya geri al.
        features: GPU tensor (1, 192, feat_h, feat_w)
        """
        pool = self.pools[model_idx]
        g    = self.gaussians[model_idx]

        # Feature'ı pool et (backbone çıktısı tam çözünürlükte kalır)
        if pool > 1:
            feats_pooled = F.avg_pool2d(features, kernel_size=pool, stride=pool)
        else:
            feats_pooled = features

        # Precision'ı GPU'ya taşı (geçici)
        g.to(self.device)
        dist_map = g.mahalanobis_map(feats_pooled)[0]   # (h, w) GPU
        dist_np  = dist_map.cpu().numpy()
        # Precision'ı tekrar CPU'ya al (VRAM serbest kalır)
        g.to("cpu")

        H, W    = self.img_size
        resized = cv2.resize(dist_np, (W, H), interpolation=cv2.INTER_LINEAR)
        score   = float(resized.mean())
        heatmap = normalize_heatmap(resized)
        return heatmap, score

    # ── Dışa açık API ────────────────────────────────────────────────

    def inspect(
        self,
        crop_id: int,
        image_path: str,
        threshold: float | None = None,
    ) -> dict:
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
        results = []
        for crop_id, image_path in crops:
            results.append(self.inspect(crop_id, image_path, threshold))
        return results