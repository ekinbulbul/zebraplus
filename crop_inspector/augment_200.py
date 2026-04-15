"""
augment_200.py
---------------
4 görüntüden translation + rotation augmentasyonu ile
tam olarak 200 adet 1200x1200 görüntü üretir.

Çıktı dosyaları sıfır dolgulu isimlendirilir:
    image_000.png  ← model_0
    image_001.png  ← model_1
    ...
    image_199.png  ← model_199

Kullanım:
    python augment_200.py `
        --input_dir   good_images/ `
        --output_dir  photos/ `
        --target_size 1200 1200 `
        --max_shift   20 `
        --max_angle   3.0 `
        --seed        42
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
TARGET_COUNT = 200


def parse_args():
    p = argparse.ArgumentParser(description="200 görüntü augmentasyonu")
    p.add_argument("--input_dir",   type=str, required=True)
    p.add_argument("--output_dir",  type=str, required=True)
    p.add_argument("--target_size", type=int, nargs=2, default=[1200, 1200],
                   metavar=("H", "W"))
    p.add_argument("--max_shift",   type=int, default=20)
    p.add_argument("--max_angle",   type=float, default=3.0)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def collect_images(input_dir):
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


def resize_to_target(img, target_h, target_w):
    h, w   = img.shape[:2]
    interp = cv2.INTER_CUBIC if (target_h > h or target_w > w) else cv2.INTER_AREA
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def translate(img, dx, dy):
    """Wrap-around kaydırma — zebra periyodunu korur."""
    shifted = np.roll(img, dy, axis=0)
    shifted = np.roll(shifted, dx, axis=1)
    return shifted


def rotate(img, angle):
    """Merkez etrafında döndür, reflect border."""
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def augment_one(img, rng, max_shift, max_angle):
    dx    = int(rng.integers(-max_shift, max_shift + 1))
    dy    = int(rng.integers(-max_shift, max_shift + 1))
    angle = float(rng.uniform(-max_angle, max_angle))
    return rotate(translate(img, dx, dy), angle)


def main():
    args       = parse_args()
    rng        = np.random.default_rng(args.seed)
    target_h, target_w = args.target_size

    input_paths = collect_images(args.input_dir)
    n_src       = len(input_paths)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Kaynak görüntü  : {n_src}")
    print(f"Hedef adet      : {TARGET_COUNT}")
    print(f"Her kaynaktan   : ~{TARGET_COUNT // n_src} varyasyon")
    print(f"Hedef boyut     : {target_h}x{target_w}")
    print(f"Max shift       : ±{args.max_shift} px")
    print(f"Max rotasyon    : ±{args.max_angle}°")
    print(f"Çıktı klasörü   : {out_dir}\n")

    # Yükle ve resize et
    sources = []
    for p in input_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [UYARI] Okunamadı: {p}")
            continue
        h_orig, w_orig = img.shape[:2]
        img = resize_to_target(img, target_h, target_w)
        print(f"  {p.name}: {w_orig}x{h_orig} → {target_w}x{target_h}")
        sources.append(img)

    if not sources:
        raise RuntimeError("Hiç kaynak görüntü yüklenemedi!")

    # Kaç tane her kaynaktan üretilecek
    per_source = TARGET_COUNT // n_src
    extras     = TARGET_COUNT % n_src   # ilk 'extras' kaynağa 1 fazla

    output_idx = 0
    for src_i, img in enumerate(sources):
        count = per_source + (1 if src_i < extras else 0)
        for _ in range(count):
            aug   = augment_one(img, rng, args.max_shift, args.max_angle)
            fname = out_dir / f"image_{output_idx:03d}.png"
            cv2.imwrite(str(fname), aug)
            output_idx += 1

    print(f"\nTamamlandı.")
    print(f"Üretilen görüntü : {output_idx}")
    print(f"Klasör           : {out_dir}")
    print(f"\nDosya isimleri image_000.png ... image_{output_idx-1:03d}.png")
    print(f"model_0 ... model_{output_idx-1} ile eşleşir.")


if __name__ == "__main__":
    main()