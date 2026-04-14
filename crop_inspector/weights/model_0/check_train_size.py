import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "weights/model_0/gaussian_model.npz"

data = np.load(path)
mean      = data["mean"]        # (C, H, W)
precision = data["precision"]   # (HW, C, C)

C  = mean.shape[0]
H, W = mean.shape[1], mean.shape[2]
HW = precision.shape[0]

print(f"Shape → mean: {mean.shape}, precision: {precision.shape}")
print(f"Kanal (C): {C}, Spatial: {H}x{W}")

# Precision'dan kovaryansı geri al (birkaç piksel örneği)
sample_indices = [0, HW//4, HW//2, 3*HW//4, HW-1]
ranks = []
for idx in sample_indices:
    P = precision[idx]           # (C, C)
    cov = np.linalg.inv(P)       # kovaryansa geri dön
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-6]
    ranks.append(len(eigvals))

print(f"\nKovaryans rank tahminleri (5 piksel): {ranks}")
print(f"Ortalama rank: {np.mean(ranks):.0f} / {C}")
print(f"\nTahmin edilen minimum eğitim görüntüsü: ~{int(np.mean(ranks)) + 1}")
print(f"(Rank = kaç bağımsız örnek gördüğünün alt sınırı)")