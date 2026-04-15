"""
visualize.py
-------------
Tek bir görüntü için heatmap üretir ve kaydeder.
Mevcut run.py'yi bozmaz, ayrı çalışır.

Kullanım:
    python visualize.py --image foto.png --weights weights_pool8/ --save_dir viz/
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from model.feature_extractor import FeatureExtractor
from model.gaussian_model    import GaussianModel
from core.transforms         import build_transform, load_batch, normalize_heatmap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",      type=str, required=True, help="Analiz edilecek görüntü")
    p.add_argument("--weights",    type=str, default="weights/", help="Model klasörü")
    p.add_argument("--model_idx",  type=int, default=0, help="Hangi model kullanılsın")
    p.add_argument("--save_dir",   type=str, default="viz/")
    p.add_argument("--img_size",   type=int, nargs=2, default=[1200, 1200])
    p.add_argument("--alpha",      type=float, default=0.5, help="Overlay opaklığı")
    p.add_argument("--device",     type=str, default=None)
    return p.parse_args()


def main():
    args     = parse_args()
    device   = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    img_size = tuple(args.img_size)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Pool bilgisini meta'dan oku
    model_dir = Path(args.weights) / f"model_{args.model_idx}"
    meta_path = model_dir / "meta.txt"
    pool = 4
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("pool="):
                pool = int(line.split("=")[1])

    print(f"[viz] Görüntü  : {args.image}")
    print(f"[viz] Model    : model_{args.model_idx}  pool=x{pool}")
    print(f"[viz] Device   : {device}")

    # Backbone
    transform = build_transform(img_size)
    backbone  = FeatureExtractor(pretrained=True, freeze=True).to(device)
    backbone.eval()

    # Gaussian
    g = GaussianModel(device=device)
    g.load(str(model_dir / "gaussian_model"))
    g.to(device)

    # Feature çıkar
    with torch.no_grad():
        tensor = load_batch([args.image], transform).to(device)
        feats  = backbone(tensor)
        if pool > 1:
            feats = F.avg_pool2d(feats, kernel_size=pool, stride=pool)
        dist_map = g.mahalanobis_map(feats)[0]  # (h, w)

    dist_np  = dist_map.cpu().numpy()
    score    = float(dist_np.mean())
    print(f"[viz] Score    : {score:.4f}")

    # Heatmap
    H, W     = img_size
    resized  = cv2.resize(dist_np, (W, H), interpolation=cv2.INTER_LINEAR)
    heatmap  = normalize_heatmap(resized)

    # Orijinal görüntüyü yükle
    img_orig = cv2.imread(args.image)
    if img_orig is None:
        raise FileNotFoundError(f"Görüntü açılamadı: {args.image}")
    img_orig = cv2.resize(img_orig, (W, H))

    # Heatmap renklendirme
    hm_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay
    overlay  = cv2.addWeighted(img_orig, 1 - args.alpha, hm_color, args.alpha, 0)

    # Score yazısı
    cv2.putText(overlay, f"score: {score:.4f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Kaydet
    stem         = Path(args.image).stem
    heatmap_path = save_dir / f"{stem}_heatmap.png"
    overlay_path = save_dir / f"{stem}_overlay.png"
    raw_path     = save_dir / f"{stem}_dist_raw.png"

    cv2.imwrite(str(heatmap_path), (heatmap * 255).astype(np.uint8))
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(raw_path),     (normalize_heatmap(dist_np) * 255).astype(np.uint8))

    print(f"[viz] Heatmap  : {heatmap_path}")
    print(f"[viz] Overlay  : {overlay_path}")
    print(f"[viz] Raw dist : {raw_path}")


if __name__ == "__main__":
    main()
