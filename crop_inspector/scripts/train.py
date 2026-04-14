"""
scripts/train.py
-----------------
Tek bir backbone ile belirtilen model index'i (0-9) için
GaussianModel eğitir ve weights/model_<idx>/ altına kaydeder.

Kullanım:
    python scripts/train.py `
        --model_idx   0 `
        --data_dir    data/train/normal `
        --weights_dir weights/ `
        --pool        4 `
        --img_size    1200 1200 `
        --batch_size  4 `
        --device      cuda

Çıktı:
    weights/model_0/gaussian_model.npz
    weights/model_0/meta.txt   ← pool, img_size, N bilgisi
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.feature_extractor import FeatureExtractor
from model.gaussian_model    import GaussianModel
from core.transforms         import build_transform, load_batch


def parse_args():
    p = argparse.ArgumentParser(description="CropInspector — Eğitim")
    p.add_argument("--model_idx",      type=int, required=True,
                   help="Hangi model eğitilecek (0-9)")
    p.add_argument("--data_dir",       type=str, required=True,
                   help="Normal (defektsiz) görüntü klasörü")
    p.add_argument("--weights_dir",    type=str, default="weights/")
    p.add_argument("--img_size",       type=int, nargs=2, default=[1200, 1200],
                   metavar=("H", "W"))
    p.add_argument("--pool",           type=int, default=4,
                   help="Spatial pooling faktörü Gaussian için (1=yok, 4=önerilen)")
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--device",         type=str, default=None)
    p.add_argument("--regularization", type=float, default=0.01)
    p.add_argument("--fp16_save",      action="store_true",
                   help="Ağırlıkları fp16 olarak kaydet (yarı boyut)")
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
    args     = parse_args()
    device   = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    img_size = tuple(args.img_size)
    save_dir = Path(args.weights_dir) / f"model_{args.model_idx}"

    feat_h = img_size[0] // 4          # backbone stride=4
    feat_w = img_size[1] // 4
    pool_h = feat_h // args.pool
    pool_w = feat_w // args.pool

    print("=" * 56)
    print(f"  CropInspector — Eğitim  [model_{args.model_idx}]")
    print("=" * 56)
    print(f"  data_dir      : {args.data_dir}")
    print(f"  save_dir      : {save_dir}")
    print(f"  img_size      : {img_size[0]}x{img_size[1]}")
    print(f"  device        : {device}")
    print(f"  pool faktörü  : x{args.pool}  ({feat_h}x{feat_w} → {pool_h}x{pool_w})")
    print(f"  fp16 kayıt    : {args.fp16_save}")
    print("=" * 56)

    train_paths = collect_images(args.data_dir, args.extensions)
    print(f"\n{len(train_paths)} görüntü bulundu.\n")

    transform = build_transform(img_size)
    backbone  = FeatureExtractor(pretrained=True, freeze=True).to(device)
    backbone.eval()

    gaussian  = GaussianModel(regularization=args.regularization, device=device)

    with torch.no_grad():
        for i in tqdm(range(0, len(train_paths), args.batch_size), desc="Feature çıkarımı"):
            batch = load_batch(train_paths[i:i + args.batch_size], transform).to(device)
            feats = backbone(batch)   # (B, 192, feat_h, feat_w)

            # Gaussian için spatial pooling
            if args.pool > 1:
                feats = F.avg_pool2d(feats, kernel_size=args.pool, stride=args.pool)

            gaussian.add_features(feats)

    print("\nGaussian model fit ediliyor...")
    gaussian.fit()

    save_dir.mkdir(parents=True, exist_ok=True)
    gaussian.save(str(save_dir / "gaussian_model"), fp16=args.fp16_save)

    # Meta bilgisi kaydet — inspector yüklerken pool faktörünü bilsin
    meta_path = save_dir / "meta.txt"
    meta_path.write_text(
        f"pool={args.pool}\n"
        f"img_size={img_size[0]},{img_size[1]}\n"
        f"n_train={len(train_paths)}\n"
        f"feat_spatial={pool_h},{pool_w}\n",
        encoding="utf-8",
    )

    print(f"\nAğırlıklar kaydedildi → {save_dir}/gaussian_model.npz")
    print(f"Meta bilgisi   → {meta_path}")


if __name__ == "__main__":
    main()