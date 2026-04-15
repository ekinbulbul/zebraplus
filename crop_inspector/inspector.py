"""
inspector.py — torch.compile ile hızlandırılmış backbone
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path
import threading
from queue import Queue

from model.feature_extractor import FeatureExtractor
from model.gaussian_model    import GaussianModel
from core.transforms         import build_transform, load_batch


def _load_meta(model_dir: Path) -> dict:
    meta = {"pool": 4, "img_size": (1200, 1200)}
    meta_path = model_dir / "meta.txt"
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

    def __init__(
        self,
        weights_dir: str,
        img_size: tuple[int, int] = (1200, 1200),
        device: str | None = None,
        pretrained: bool = True,
        prefetch_ahead: int = 2,
        use_compile: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device         = torch.device(device)
        self.img_size       = img_size
        self.weights_dir    = Path(weights_dir)
        self.transform      = build_transform(img_size)
        self.prefetch_ahead = prefetch_ahead

        print(f"[CropInspector] Backbone yükleniyor → device={self.device}")
        backbone = FeatureExtractor(pretrained=True, freeze=True).to(self.device)
        backbone.eval()

        if use_compile and hasattr(torch, "compile"):
            print("[CropInspector] torch.compile uygulanıyor (ilk crop yavaş olacak)...")
            self.backbone = torch.compile(backbone, mode="reduce-overhead")
        else:
            self.backbone = backbone

        print(f"[CropInspector] Hazır. prefetch_ahead={prefetch_ahead}")

    def _model_pt_path(self, model_idx: int) -> str:
        return str(self.weights_dir / f"model_{model_idx}" / "gaussian_model.pt")

    def _load_data(self, model_idx: int) -> dict:
        return torch.load(self._model_pt_path(model_idx), weights_only=True)

    def _make_gaussian(self, data: dict, model_idx: int) -> tuple[GaussianModel, int]:
        model_dir  = self.weights_dir / f"model_{model_idx}"
        meta       = _load_meta(model_dir)
        g          = GaussianModel(device="cpu")
        g.mean     = data["mean"]
        g.precision= data["precision"]
        g._spatial = data["spatial"]
        return g, meta["pool"]

    @torch.no_grad()
    def _extract(self, image_path: str, pool: int) -> torch.Tensor:
        tensor = load_batch([image_path], self.transform).to(self.device)
        feats  = self.backbone(tensor)
        if pool > 1:
            feats = F.avg_pool2d(feats, kernel_size=pool, stride=pool)
        return feats

    def _score(self, features: torch.Tensor, gaussian: GaussianModel) -> float:
        gaussian.to(self.device)
        with torch.no_grad():
            dist_map = gaussian.mahalanobis_map(features)[0]
        score = float(dist_map.mean().cpu())
        del gaussian
        return score

    def inspect_batch(
        self,
        crops: list[tuple[int, str]],
        threshold: float | None = None,
    ) -> list[dict]:
        if not crops:
            return []

        queue: Queue = Queue(maxsize=self.prefetch_ahead + 1)

        def loader_worker():
            for crop_id, _ in crops:
                try:
                    data = self._load_data(crop_id)
                    queue.put((crop_id, data))
                except Exception as e:
                    queue.put((crop_id, e))

        loader = threading.Thread(target=loader_worker, daemon=True)
        loader.start()

        results = []
        for crop_id, image_path in crops:
            queued_id, data = queue.get()
            assert queued_id == crop_id

            if isinstance(data, Exception):
                raise data

            gaussian, pool = self._make_gaussian(data, crop_id)
            features       = self._extract(image_path, pool)
            score          = self._score(features, gaussian)

            label = "UNKNOWN"
            if threshold is not None:
                label = "DEFECT" if score >= threshold else "NORMAL"

            results.append({
                "crop_id"   : crop_id,
                "model_idx" : crop_id,
                "score"     : round(score, 4),
                "label"     : label,
            })

        loader.join()
        return results

    def inspect(
        self,
        crop_id: int,
        image_path: str,
        threshold: float | None = None,
    ) -> dict:
        data           = self._load_data(crop_id)
        gaussian, pool = self._make_gaussian(data, crop_id)
        features       = self._extract(image_path, pool)
        score          = self._score(features, gaussian)

        label = "UNKNOWN"
        if threshold is not None:
            label = "DEFECT" if score >= threshold else "NORMAL"

        return {
            "crop_id"   : crop_id,
            "model_idx" : crop_id,
            "score"     : round(score, 4),
            "label"     : label,
        }