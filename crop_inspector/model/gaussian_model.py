"""
gaussian_model.py
------------------
Her uzamsal konum (h, w) için feature vektörlerinin
Gaussian dağılımını öğrenir ve Mahalanobis mesafesi hesaplar.

fit()       : CPU'da chunk chunk çalışır — VRAM sorunu olmaz
save/load   : fp16 destekli
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path


class GaussianModel:

    def __init__(self, regularization: float = 0.01, device: str | torch.device = "cpu"):
        self.regularization = regularization
        self.device = torch.device(device)

        self.mean: torch.Tensor | None = None
        self.precision: torch.Tensor | None = None
        self._spatial: tuple[int, int] | None = None

        self._bank: list[torch.Tensor] = []

    def add_features(self, features: torch.Tensor | np.ndarray):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        assert features.ndim == 4, \
            f"Beklenen: (B,C,H,W), gelen: {features.shape}"
        # CPU'ya al — bank CPU'da tutulur, VRAM baskısı olmaz
        self._bank.append(features.float().cpu())

    def fit(self, chunk_size: int = 500):
        """
        Kovaryans ve matris inversiyonu tamamen CPU'da yapılır.
        VRAM kullanmaz.

        chunk_size : kaç spatial konum aynı anda işlensin
                     16 GB RAM → 500
                      8 GB RAM → 200
        """
        assert len(self._bank) > 0, "Önce add_features() ile veri ekle!"

        bank = torch.cat(self._bank, dim=0)   # (N, C, H, W) — CPU
        N, C, H, W = bank.shape
        print(f"[GaussianModel.fit] N={N} görüntü, C={C}, H={H}, W={W}")
        print(f"[GaussianModel.fit] CPU'da chunk_size={chunk_size} ile işleniyor...")

        hw        = H * W
        bank_flat = bank.permute(2, 3, 0, 1).reshape(hw, N, C)  # (hw, N, C)
        mean_flat = bank_flat.mean(dim=1)                         # (hw, C)

        precision = torch.zeros(hw, C, C, dtype=torch.float32)
        reg       = self.regularization * torch.eye(C, dtype=torch.float32)

        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, hw, chunk_size), desc="Gaussian fit (CPU)")
        except ImportError:
            iterator = range(0, hw, chunk_size)

        for start in iterator:
            end      = min(start + chunk_size, hw)
            chunk    = bank_flat[start:end]                       # (k, N, C)
            mean_c   = mean_flat[start:end]                       # (k, C)
            centered = chunk - mean_c.unsqueeze(1)                # (k, N, C)
            cov      = torch.bmm(
                centered.transpose(1, 2), centered
            ) / max(N - 1, 1)                                     # (k, C, C)
            cov     += reg.unsqueeze(0)
            precision[start:end] = torch.linalg.inv(cov)

        self.mean      = mean_flat.T.reshape(C, H, W)            # (C, H, W)
        self.precision = precision                                 # (hw, C, C)
        self._spatial  = (H, W)

        self._bank.clear()
        print(f"[GaussianModel.fit] Tamamlandı. "
              f"mean={self.mean.shape}, precision={self.precision.shape}")

    def mahalanobis_map(self, features: torch.Tensor | np.ndarray) -> torch.Tensor:
        self._check_fitted()

        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        features = features.float().to(self.device)

        B, C, H, W = features.shape
        self._check_shape(C, H, W)
        hw = H * W

        diff      = features - self.mean.to(self.device).unsqueeze(0)
        diff_flat = diff.permute(0, 2, 3, 1).reshape(B, hw, C)

        P   = self.precision.to(self.device).unsqueeze(0).expand(B, -1, -1, -1)
        d   = diff_flat.unsqueeze(-1)
        Pd  = torch.matmul(P, d).squeeze(-1)

        score_flat = (diff_flat * Pd).sum(dim=-1)
        dist_map   = score_flat.clamp(min=0.0).sqrt().reshape(B, H, W)
        return dist_map

    def save(self, path: str, fp16: bool = False):
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dtype = np.float16 if fp16 else np.float32
        np.savez_compressed(
            path,
            mean=self.mean.cpu().numpy().astype(dtype),
            precision=self.precision.cpu().numpy().astype(dtype),
            spatial=np.array(self._spatial),
            fp16=np.array(fp16),
        )
        tag = "fp16" if fp16 else "fp32"
        print(f"[GaussianModel] Kaydedildi ({tag}) → {path}.npz")

    def load(self, path: str):
        fpath = path if path.endswith(".npz") else path + ".npz"
        data  = np.load(fpath)
        self.mean      = torch.from_numpy(data["mean"].astype(np.float32))
        self.precision = torch.from_numpy(data["precision"].astype(np.float32))
        self._spatial  = tuple(data["spatial"].tolist())
        print(f"[GaussianModel] Yüklendi ← {fpath} | "
              f"mean={self.mean.shape}, precision={self.precision.shape}")

    def to(self, device: str | torch.device) -> "GaussianModel":
        self.device = torch.device(device)
        if self.mean is not None:
            self.mean = self.mean.to(self.device)
        if self.precision is not None:
            self.precision = self.precision.to(self.device)
        return self

    def is_fitted(self) -> bool:
        return self.mean is not None and self.precision is not None

    def _check_fitted(self):
        if not self.is_fitted():
            raise RuntimeError("Model henüz fit edilmemiş. Önce fit() çağır.")

    def _check_shape(self, C: int, H: int, W: int):
        exp_C, exp_H, exp_W = self.mean.shape
        if (C, H, W) != (exp_C, exp_H, exp_W):
            raise ValueError(
                f"Feature shape ({C},{H},{W}) gaussian shape "
                f"({exp_C},{exp_H},{exp_W}) ile uyumsuz."
            )