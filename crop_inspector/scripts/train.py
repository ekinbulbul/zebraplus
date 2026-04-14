"""
scripts/train.py
-----------------
Tek bir backbone ile belirtilen model index'i (0-9) için
GaussianModel eğitir ve weights/model_<idx>/ altına kaydeder.

Backbone paylaşıldığı için her model kendi veri klasöründen
ayrı ayrı eğitilir; backbone bir kez yüklenir, feature çıkarımı
yapılır ve Gaussian fit edilir.

Kullanım — model 3'ü eğit:
    python scripts/train.py \
        --model_idx  3 \
        --data_dir   data/train/model_3/normal \
        --weights_dir weights/ \
        --img_size   256 256 \
        --batch_size 8 \
        --device     cuda

Ağırlık çıktısı:
    weights/model_3/gaussian_model.npz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from model.feature_extractor import FeatureExtractor
from model.gaussian_model    import GaussianModel
from core.transforms         import build_transform, load_batch


def parse_args():
    p = argparse.ArgumentParser(description="CropInspector — Eğitim")
    p.add_argument("--model_idx",      type=int, required=True,
                   help="Hangi model eğitilecek (0-9)")
    p.add_argument("--data_dir",       type=str, required=True,
                   help="Bu modele ait normal (defektsiz) görüntü klasörü")
    p.add_argument("--weights_dir",    type=str, default="weights/",
                   help="Tüm modellerin ağırlıklarının tutulduğu ana klasör")
    p.add_argument("--img_size",       type=int, nargs=2, default=[256, 256])
    p.add_argument("--batch_size",     type=int, default=8)
    p.add_argument("--device",         type=str, default=None)
    p.add_argument("--regularization", type=float, default=0.01)
    p.add_argument("--extensions",     type=str, nargs="+",
                   default=[".png", ".jpg", ".jpeg", ".bmp"])
    return p.parse_args()


def collect_images(data_dir, extensions):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Klasör bulunamadı: {data_dir}")
    seen, paths = set(), []
    for ext in extensions:
        for p in sorted(data_path.glob(f"*{ext}")) + sorted(data_path.glob(f"*{ext.upper()}")):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                paths.append(str(p))
    if not paths:
        raise ValueError(f"'{data_dir}' içinde görüntü bulunamadı!")
    return paths


def main():
    args   = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    img_size = tuple(args.img_size)

    save_dir = Path(args.weights_dir) / f"model_{args.model_idx}"

    print("=" * 52)
    print(f"  CropInspector — Eğitim  [model_{args.model_idx}]")
    print("=" * 52)
    print(f"  data_dir    : {args.data_dir}")
    print(f"  save_dir    : {save_dir}")
    print(f"  img_size    : {img_size}")
    print(f"  device      : {device}")
    print("=" * 52)

    train_paths = collect_images(args.data_dir, args.extensions)
    print(f"\n{len(train_paths)} görüntü bulundu.\n")

    # Backbone: paylaşılan ağırlıklar, sadece feature çıkarımı için kullanılır
    transform = build_transform(img_size)
    backbone  = FeatureExtractor(pretrained=True, freeze=True).to(device)
    backbone.eval()

    # Bu modele özgü Gaussian
    gaussian  = GaussianModel(regularization=args.regularization, device=device)

    with torch.no_grad():
        for i in tqdm(range(0, len(train_paths), args.batch_size), desc="Feature çıkarımı"):
            batch = load_batch(train_paths[i:i + args.batch_size], transform).to(device)
            feats = backbone(batch)
            gaussian.add_features(feats)

    print("\nGaussian model fit ediliyor...")
    gaussian.fit()

    save_dir.mkdir(parents=True, exist_ok=True)
    gaussian.save(str(save_dir / "gaussian_model"))
    print(f"\nAğırlıklar kaydedildi → {save_dir}/gaussian_model.npz")


if __name__ == "__main__":
    main()
