"""
gaussian_model.py
------------------
torch.save/load formatı — npz'den 4x daha hızlı yükleme.
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
        assert features.ndim == 4, f"Beklenen: (B,C,H,W), gelen: {features.shape}"
        self._bank.append(features.float().cpu())

    def fit(self, chunk_size: int = 500):
        assert len(self._bank) > 0, "Önce add_features() ile veri ekle!"

        bank = torch.cat(self._bank, dim=0)
        N, C, H, W = bank.shape
        print(f"[GaussianModel.fit] N={N} görüntü, C={C}, H={H}, W={W}")
        print(f"[GaussianModel.fit] CPU'da chunk_size={chunk_size} ile işleniyor...")

        hw        = H * W
        bank_flat = bank.permute(2, 3, 0, 1).reshape(hw, N, C)
        mean_flat = bank_flat.mean(dim=1)

        precision = torch.zeros(hw, C, C, dtype=torch.float32)
        reg       = self.regularization * torch.eye(C, dtype=torch.float32)

        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, hw, chunk_size), desc="Gaussian fit (CPU)")
        except ImportError:
            iterator = range(0, hw, chunk_size)

        for start in iterator:
            end      = min(start + chunk_size, hw)
            chunk    = bank_flat[start:end]
            mean_c   = mean_flat[start:end]
            centered = chunk - mean_c.unsqueeze(1)
            cov      = torch.bmm(centered.transpose(1, 2), centered) / max(N - 1, 1)
            cov     += reg.unsqueeze(0)
            precision[start:end] = torch.linalg.inv(cov)

        self.mean      = mean_flat.T.reshape(C, H, W)
        self.precision = precision
        self._spatial  = (H, W)
        self._bank.clear()
        print(f"[GaussianModel.fit] Tamamlandı. mean={self.mean.shape}, precision={self.precision.shape}")

    def mahalanobis_map(self, features: torch.Tensor | np.ndarray) -> torch.Tensor:
        self._check_fitted()

        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        features  = features.float().to(self.device)
        mean      = self.mean.float().to(self.device)
        precision = self.precision.float().to(self.device)

        B, C, H, W = features.shape
        hw = H * W

        diff      = features - mean.unsqueeze(0)
        diff_flat = diff.permute(0, 2, 3, 1).reshape(B, hw, C)

        P   = precision.unsqueeze(0).expand(B, -1, -1, -1)
        d   = diff_flat.unsqueeze(-1)
        Pd  = torch.matmul(P, d).squeeze(-1)

        score_flat = (diff_flat * Pd).sum(dim=-1)
        dist_map   = score_flat.clamp(min=0.0).sqrt().reshape(B, H, W)
        return dist_map

    def save(self, path: str, fp16: bool = True):
        """torch.save formatında kaydet (.pt)"""
        self._check_fitted()
        pt_path = Path(path).with_suffix('.pt')
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        dtype = torch.float16 if fp16 else torch.float32
        torch.save({
            'mean'     : self.mean.cpu().to(dtype),
            'precision': self.precision.cpu().to(dtype),
            'spatial'  : self._spatial,
            'fp16'     : fp16,
        }, str(pt_path))
        print(f"[GaussianModel] Kaydedildi ({'fp16' if fp16 else 'fp32'}) → {pt_path}")

    def load(self, path: str):
        """
        .pt veya .npz formatını otomatik tanır.
        .pt → torch.load (hızlı)
        .npz → np.load (eski format, yavaş)
        """
        # .pt öncelikli
        pt_path  = Path(path).with_suffix('.pt')
        npz_path = Path(path).with_suffix('.npz')

        if pt_path.exists():
            data           = torch.load(str(pt_path), weights_only=True)
            self.mean      = data['mean']
            self.precision = data['precision']
            self._spatial  = data['spatial']
            print(f"[GaussianModel] Yüklendi (pt, {self.mean.dtype}) ← {pt_path} | "
                  f"mean={self.mean.shape}, precision={self.precision.shape}")
        elif npz_path.exists():
            data           = np.load(str(npz_path))
            self.mean      = torch.from_numpy(np.array(data['mean']))
            self.precision = torch.from_numpy(np.array(data['precision']))
            self._spatial  = tuple(data['spatial'].tolist())
            print(f"[GaussianModel] Yüklendi (npz, {self.mean.dtype}) ← {npz_path} | "
                  f"mean={self.mean.shape}, precision={self.precision.shape}")
        else:
            raise FileNotFoundError(f"Model bulunamadı: {pt_path} veya {npz_path}")

        self.mean      = self.mean.to(self.device)
        self.precision = self.precision.to(self.device)

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