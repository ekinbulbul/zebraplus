"""
Farklı kaydetme formatlarının yükleme hızını karşılaştır.
"""
import time
import numpy as np
import torch
from pathlib import Path

# Mevcut npz'yi yükle
d         = np.load('weights/model_0/gaussian_model.npz')
precision = torch.from_numpy(np.array(d['precision']))  # f16 tensor
mean      = torch.from_numpy(np.array(d['mean']))

print(f"precision shape: {precision.shape}  dtype: {precision.dtype}  size: {precision.nbytes/1e6:.0f} MB")
print()

# 1. torch.save formatına dönüştür
torch.save({'mean': mean, 'precision': precision}, 'weights/model_0/gaussian_model.pt')
pt_size = Path('weights/model_0/gaussian_model.pt').stat().st_size / 1e6
print(f"torch.pt boyutu: {pt_size:.0f} MB")

# 2. Hız karşılaştırması
REPEAT = 5

# npz yükleme
times_npz = []
for _ in range(REPEAT):
    t0 = time.perf_counter()
    d2 = np.load('weights/model_0/gaussian_model.npz')
    p2 = torch.from_numpy(np.array(d2['precision']))
    times_npz.append((time.perf_counter() - t0) * 1000)
print(f"npz yükleme    : ort={sum(times_npz)/REPEAT:.0f}ms  min={min(times_npz):.0f}ms")

# torch.load yükleme
times_pt = []
for _ in range(REPEAT):
    t0 = time.perf_counter()
    data = torch.load('weights/model_0/gaussian_model.pt', weights_only=True)
    p3   = data['precision']
    times_pt.append((time.perf_counter() - t0) * 1000)
print(f"torch.pt yükleme: ort={sum(times_pt)/REPEAT:.0f}ms  min={min(times_pt):.0f}ms")

print()
print(f"Kazanç: {sum(times_npz)/sum(times_pt):.1f}x daha hızlı")
