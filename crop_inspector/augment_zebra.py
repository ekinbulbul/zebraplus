"""
augment_zebra.py
-----------------
6 good görüntüden translation + rotation augmentasyonuyla
1000 eğitim görüntüsü üretir.

Görüntüler önce --target_size'a yeniden boyutlandırılır,
ardından augmentasyon uygulanır.

Kullanım:
    python augment_zebra.py \
        --input_dir   good_images/ \
        --output_dir  data/train/normal/ \
        --target_size 1200 1200 \
        --target      1000 \
        --max_shift   20 \
        --max_angle   3.0 \
        --seed        42
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]


def parse_args():
    p = argparse.ArgumentParser(description="Zebra Augmentation")
    p.add_argument("--input_dir",   type=str, required=True)
    p.add_argument("--output_dir",  type=str, required=True)
    p.add_argument("--target_size", type=int, nargs=2, default=[1200, 1200],
                   metavar=("H", "W"),
                   help="Çıktı boyutu (default: 1200 1200)")
    p.add_argument("--target",      type=int, default=1000)
    p.add_argument("--max_shift",   type=int, default=20,
                   help="Maksimum kaydırma (piksel). Zebra çizgi aralığının yarısını geçme.")
    p.add_argument("--max_angle",   type=float, default=3.0,
                   help="Maksimum rotasyon açısı (derece). 2-3 önerilir.")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def collect_images(input_dir: str) -> list[Path]:
    d = Path(input_dir)
    if not d.exists():
        raise FileNotFoundError(f"Klasör bulunamadı: {input_dir}")
    seen, paths = set(), []
    for ext in EXTENSIONS:
        for p in sorted(d.glob(f"*{ext}")) + sorted(d.glob(f"*{ext.upper()}")):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                paths.append(p)
    if not paths:
        raise ValueError(f"'{input_dir}' içinde görüntü bulunamadı!")
    return paths


def resize_to_target(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Görüntüyü hedef boyuta getirir.
    Büyütmek için INTER_CUBIC kullanılır — zebra çizgilerini
    INTER_LINEAR'dan daha keskin korur, alias yaratmaz.
    Küçültmek için INTER_AREA kullanılır — en kaliteli downsample.
    """
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img

    interp = cv2.INTER_CUBIC if (target_h > h or target_w > w) else cv2.INTER_AREA
    resized = cv2.resize(img, (target_w, target_h), interpolation=interp)

    scale_h = target_h / h
    scale_w = target_w / w
    return resized


def translate(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Görüntüyü (dx, dy) kadar kaydırır.
    Wrap-around: zebra yansımasının periyodik yapısını korur —
    kaydırma sonucu çıkan boşluk, çizgilerin devamıyla dolar.
    """
    shifted = np.roll(img, dy, axis=0)
    shifted = np.roll(shifted, dx, axis=1)
    return shifted


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Görüntüyü merkez etrafında döndürür.
    BORDER_REFLECT_101: kenarı ayna gibi yansıtır, siyah kenar oluşmaz.
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale=1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def augment_one(img: np.ndarray, rng: np.random.Generator,
                max_shift: int, max_angle: float) -> np.ndarray:
    dx    = int(rng.integers(-max_shift, max_shift + 1))
    dy    = int(rng.integers(-max_shift, max_shift + 1))
    angle = float(rng.uniform(-max_angle, max_angle))
    out   = translate(img, dx, dy)
    out   = rotate(out, angle)
    return out


def main():
    args     = parse_args()
    rng      = np.random.default_rng(args.seed)
    target_h, target_w = args.target_size

    input_paths = collect_images(args.input_dir)
    n_src       = len(input_paths)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Kaynak görüntü  : {n_src}")
    print(f"Hedef boyut     : {target_h}x{target_w}")
    print(f"Hedef adet      : {args.target}")
    print(f"Her kaynaktan   : ~{args.target // n_src} varyasyon")
    print(f"Max shift       : ±{args.max_shift} px")
    print(f"Max rotasyon    : ±{args.max_angle}°")
    print(f"Çıktı klasörü   : {out_dir}\n")

    # Yükle ve resize et
    sources = []
    for p in input_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [UYARI] Okunamadı, atlanıyor: {p}")
            continue
        h_orig, w_orig = img.shape[:2]
        img_resized = resize_to_target(img, target_h, target_w)
        print(f"  {p.name}: {w_orig}x{h_orig} → {target_w}x{target_h}")
        sources.append((p.stem, img_resized))

    if not sources:
        raise RuntimeError("Hiç kaynak görüntü yüklenemedi!")

    # Orijinalleri (resize edilmiş haliyle) kaydet
    for stem, img in sources:
        cv2.imwrite(str(out_dir / f"orig_{stem}.png"), img)

    produced   = len(sources)
    remaining  = args.target - produced
    per_source = remaining // n_src
    extra      = remaining % n_src

    global_idx = 0
    for src_i, (stem, img) in enumerate(tqdm(sources, desc="Augmentasyon")):
        count = per_source + (1 if src_i < extra else 0)
        for _ in range(count):
            aug   = augment_one(img, rng, args.max_shift, args.max_angle)
            fname = out_dir / f"aug_{stem}_{global_idx:05d}.png"
            cv2.imwrite(str(fname), aug)
            global_idx += 1

    total = len(list(out_dir.glob("*.png")))
    print(f"\nTamamlandı. Üretilen toplam görüntü: {total}")
    print(f"Klasör: {out_dir}")


if __name__ == "__main__":
    main()