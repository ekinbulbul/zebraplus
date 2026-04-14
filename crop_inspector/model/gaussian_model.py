"""
gaussian_model.py
------------------
Her uzamsal konum (h, w) için feature vektörlerinin
Gaussian dağılımını öğrenir ve Mahalanobis mesafesi hesaplar.

Matematiksel temel:
    f(x,y) ~ N( μ(x,y), Σ(x,y) )

    Mahalanobis distance:
        d(f) = sqrt( (f - μ)^T  Σ^{-1}  (f - μ) )

Saklanan değerler:
    mean      : (C, H, W)       — her konumun ortalama feature vektörü
    precision : (H*W, C, C)     — her konumun precision matrisi (Σ^{-1})

GPU optimizasyonları:
    - fit()             : tamamen vektörize, torch.linalg.inv ile GPU'da toplu matris inversiyonu
    - mahalanobis_map() : saf torch operasyonları, GPU'da çalışır
    - add_features()    : GPU tensor'ları doğrudan kabul eder (CPU'ya geçiş yok)
"""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path


class GaussianModel:

    def __init__(self, regularization: float = 0.01, device: str | torch.device = "cpu"):
        self.regularization = regularization
        self.device = torch.device(device)

        self.mean: torch.Tensor | None = None       # (C, H, W)
        self.precision: torch.Tensor | None = None  # (H*W, C, C)
        self._spatial: tuple[int, int] | None = None

        self._bank: list[torch.Tensor] = []

    # ────────────────────────────────────────────────────────────────────
    # Eğitim
    # ────────────────────────────────────────────────────────────────────

    def add_features(self, features: torch.Tensor | np.ndarray):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        assert features.ndim == 4, \
            f"Beklenen: (B,C,H,W), gelen: {features.shape}"
        self._bank.append(features.float().to(self.device))

    def fit(self):
        assert len(self._bank) > 0, "Önce add_features() ile veri ekle!"

        bank = torch.cat(self._bank, dim=0)   # (N, C, H, W)
        N, C, H, W = bank.shape
        print(f"[GaussianModel.fit] N={N} görüntü, C={C}, H={H}, W={W}")

        hw = H * W

        # (N, C, H, W) → (hw, N, C)
        bank_flat = bank.permute(2, 3, 0, 1).reshape(hw, N, C)

        # Mean: (hw, C)
        mean_flat = bank_flat.mean(dim=1)

        # Vektörize covariance: (hw, C, C)
        centered = bank_flat - mean_flat.unsqueeze(1)
        cov = torch.bmm(centered.transpose(1, 2), centered) / max(N - 1, 1)

        # Regularization
        reg = self.regularization * torch.eye(C, device=self.device).unsqueeze(0)
        cov = cov + reg

        # Toplu matris inversiyonu — GPU'da paralel
        precision = torch.linalg.inv(cov)           # (hw, C, C)

        # mean: (hw, C) → (C, H, W)
        self.mean = mean_flat.T.reshape(C, H, W)
        self.precision = precision
        self._spatial = (H, W)

        self._bank.clear()
        print(f"[GaussianModel.fit] Tamamlandı. "
              f"mean={self.mean.shape}, precision={self.precision.shape}")

    # ────────────────────────────────────────────────────────────────────
    # Anomaly Scoring
    # ────────────────────────────────────────────────────────────────────

    def mahalanobis_map(self, features: torch.Tensor | np.ndarray,
                        chunk_size: int = 4096) -> torch.Tensor:
        self._check_fitted()

        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        features = features.to(dtype=self.precision.dtype, device=self.device)

        B, C, H, W = features.shape
        self._check_shape(C, H, W)
        hw = H * W

        # diff: (B, C, H, W) → (B, hw, C)
        diff = features - self.mean.unsqueeze(0)
        diff_flat = diff.permute(0, 2, 3, 1).reshape(B, hw, C)

        score_flat = torch.zeros(B, hw, device=self.device, dtype=self.precision.dtype)

        for start in range(0, hw, chunk_size):
            end = min(start + chunk_size, hw)
            clen = end - start

            P_c = self.precision[start:end]                          # (clen, C, C)
            d_c = diff_flat[:, start:end, :]                         # (B, clen, C)

            P_exp = P_c.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * clen, C, C)
            d_exp = d_c.reshape(B * clen, C, 1)
            Pd_c  = torch.bmm(P_exp, d_exp).reshape(B, clen, C)     # (B, clen, C)

            score_flat[:, start:end] = (d_c * Pd_c).sum(dim=-1)

        dist_map = score_flat.clamp(min=0.0).sqrt().reshape(B, H, W)
        return dist_map

    # ────────────────────────────────────────────────────────────────────
    # Kaydet / Yükle
    # ────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean.cpu().numpy(),
            precision=self.precision.cpu().numpy(),
            spatial=np.array(self._spatial),
        )
        print(f"[GaussianModel] Kaydedildi → {path}.npz")

    def load(self, path: str, half: bool = False):
        fpath = path if path.endswith(".npz") else path + ".npz"
        data  = np.load(fpath)
        dtype = torch.float16 if half else torch.float32

        self.mean = torch.from_numpy(data["mean"]).to(dtype=dtype, device=self.device)
        prec      = torch.from_numpy(data["precision"])
        # Eski format: (C, C, H, W) → yeni format: (H*W, C, C)
        if prec.ndim == 4 and prec.shape[0] == prec.shape[1]:
            C, _, H, W = prec.shape
            prec = prec.permute(2, 3, 0, 1).reshape(H * W, C, C)
        self.precision = prec.to(dtype=dtype, device=self.device)
        if "spatial" in data:
            self._spatial = tuple(data["spatial"].tolist())
        else:
            self._spatial = (self.mean.shape[1], self.mean.shape[2])
        print(f"[GaussianModel] Yüklendi ← {fpath} | "
              f"mean={self.mean.shape}, precision={self.precision.shape}")

    # ────────────────────────────────────────────────────────────────────
    # Yardımcılar
    # ────────────────────────────────────────────────────────────────────

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