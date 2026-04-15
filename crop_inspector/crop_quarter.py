"""
crop_quarter.py
----------------
1200x1200 görüntüleri tam ortadan böler → 4 adet 600x600 crop.

Crop düzeni:
    +----------+----------+
    | crop_0   | crop_1   |  ← sol üst, sağ üst
    | (0,0)    | (600,0)  |
    +----------+----------+
    | crop_2   | crop_3   |  ← sol alt, sağ alt
    | (0,600)  | (600,600)|
    +----------+----------+

Global crop ID:
    görüntü_0  → crop_0,  crop_1,  crop_2,  crop_3
    görüntü_1  → crop_4,  crop_5,  crop_6,  crop_7
    ...
    görüntü_74 → crop_296, crop_297, crop_298, crop_299

Kullanım:
    python crop_quarter.py --image_dir photos/ --output_dir crops_300/
"""

import argparse
import cv2
from pathlib import Path

EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]

# Sol üst, sağ üst, sol alt, sağ alt
QUARTERS = [
    (0,   0,   600, 600),  # crop_0: x_start, y_start, x_end, y_end
    (600, 0,   1200, 600),  # crop_1
    (0,   600, 600, 1200),  # crop_2
    (600, 600, 1200, 1200), # crop_3
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",  type=str, required=True)
    p.add_argument("--output_dir", type=str, default="crops_300/")
    return p.parse_args()


def collect_images(image_dir):
    d = Path(image_dir)
    seen, paths = set(), []
    for ext in EXTENSIONS:
        for p in sorted(d.glob(f"*{ext}")) + sorted(d.glob(f"*{ext.upper()}")):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                paths.append(p)
    return paths


def main():
    args      = parse_args()
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.image_dir)
    print(f"{len(images)} görüntü bulundu.")
    print(f"Toplam crop: {len(images) * 4}\n")

    global_id = 0
    for img_idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [UYARI] Okunamadı: {img_path}")
            global_id += 4
            continue

        h, w = img.shape[:2]
        if h != 1200 or w != 1200:
            print(f"  [UYARI] {img_path.name} boyutu {w}x{h} — 1200x1200 bekleniyor, atlanıyor.")
            global_id += 4
            continue

        for q_idx, (x1, y1, x2, y2) in enumerate(QUARTERS):
            crop  = img[y1:y2, x1:x2]
            fname = out_dir / f"crop_{global_id:04d}.png"
            cv2.imwrite(str(fname), crop)
            global_id += 1

        if (img_idx + 1) % 10 == 0:
            print(f"  {img_idx+1}/{len(images)} görüntü işlendi...")

    print(f"\nTamamlandı. {global_id} crop → {out_dir}")
    print(f"\nCrop - Model eşleşmesi:")
    print(f"  crop_0000..0003 → model_0..3   (görüntü_0)")
    print(f"  crop_0004..0007 → model_4..7   (görüntü_1)")
    print(f"  ...")
    print(f"  crop_{(len(images)-1)*4:04d}..{len(images)*4-1:04d} → model_{(len(images)-1)*4}..{len(images)*4-1}  (görüntü_{len(images)-1})")


if __name__ == "__main__":
    main()
