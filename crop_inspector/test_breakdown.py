"""
Her adımı ayrı ayrı ölçer.
"""
import time, sys, glob, torch
import torch.nn.functional as F
sys.path.insert(0, '.')

from model.feature_extractor import FeatureExtractor
from model.gaussian_model    import GaussianModel
from core.transforms         import build_transform, load_batch

device    = torch.device("cuda")
transform = build_transform((1200, 1200))
backbone  = FeatureExtractor(pretrained=True, freeze=True).to(device)
backbone.eval()

imgs = sorted(glob.glob('photos/*.png') + glob.glob('photos/*.bmp') + glob.glob('photos/*.jpg'))
img  = imgs[1]  # warmup için 0'ı atla

# Warmup
with torch.no_grad():
    backbone(load_batch([img], transform).to(device))

REPEAT = 5
print(f"Görüntü: {img}\n")

def measure(label, fn, repeat=REPEAT):
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    print(f"  {label:35} : {avg:7.1f} ms")
    return avg

# 1. Görüntü okuma + transform
raw = None
def read_img():
    global raw
    raw = load_batch([img], transform)
measure("Görüntü oku + transform", read_img)

# 2. CPU → GPU taşıma (görüntü)
tensor = load_batch([img], transform)
measure("Görüntü CPU → GPU", lambda: tensor.to(device))

# 3. Backbone forward
tensor_gpu = tensor.to(device)
with torch.no_grad():
    measure("Backbone forward", lambda: backbone(tensor_gpu))

# 4. Pool
feats = backbone(tensor_gpu)
measure("AvgPool x4", lambda: F.avg_pool2d(feats, 4, 4))

# 5. Model yükleme (disk → RAM)
measure("Model disk → RAM (torch.load)", 
        lambda: torch.load('weights/model_1/gaussian_model.pt', weights_only=True))

# 6. RAM → GPU (precision)
data = torch.load('weights/model_1/gaussian_model.pt', weights_only=True)
prec = data['precision']
measure("Precision RAM → GPU", lambda: prec.to(device))

# 7. Mahalanobis
g = GaussianModel(device=device)
g.load('weights/model_1/gaussian_model')
feats_pooled = F.avg_pool2d(feats, 4, 4)
with torch.no_grad():
    measure("Mahalanobis hesabı", lambda: g.mahalanobis_map(feats_pooled))

# 8. del gaussian
def del_g():
    g2 = GaussianModel(device="cpu")
    g2.load('weights/model_1/gaussian_model')
    g2.to(device)
    del g2
measure("Load + to(gpu) + del", del_g)

print()
print("Toplam tahmini (oku+gpu+backbone+pool+load+transfer+mahal):")
