"""
scripts/run.py — prefetch destekli
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspector import CropInspector

EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--weights",   type=str, default="weights/")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--save_dir",  type=str, default="results/")
    p.add_argument("--device",    type=str, default=None)
    p.add_argument("--img_size",  type=int, nargs=2, default=[1200, 1200])
    return p.parse_args()


def collect_images(image_dir):
    d = Path(image_dir)
    if not d.exists():
        raise FileNotFoundError(f"Klasör bulunamadı: {image_dir}")
    seen, paths = set(), []
    for ext in EXTENSIONS:
        for p in sorted(d.glob(f"*{ext}")) + sorted(d.glob(f"*{ext.upper()}")):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                paths.append(p)
    if not paths:
        raise ValueError(f"'{image_dir}' içinde görüntü bulunamadı!")
    return paths


def main():
    args        = parse_args()
    image_paths = collect_images(args.image_dir)
    save_dir    = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {len(image_paths)} görüntü bulundu.")
    if len(image_paths) > 200:
        image_paths = image_paths[:200]

    inspector = CropInspector(
        weights_dir=args.weights,
        img_size=tuple(args.img_size),
        device=args.device,
    )

    # Warmup — CUDA ve disk cache ısıt
    print("[run] Warmup...")
    inspector.inspect(0, str(image_paths[0]), threshold=None)

    # Tüm crop'ları batch olarak işle (prefetch aktif)
    crops = [(idx, str(p)) for idx, p in enumerate(image_paths)]

    print(f"\n  {'ID':>4}  {'Durum':^8}  {'Score':>8}  Dosya")
    print("  " + "─" * 55)

    total_start = time.perf_counter()
    results     = inspector.inspect_batch(crops, threshold=args.threshold)
    total_elapsed = time.perf_counter() - total_start

    for r, img_path in zip(results, image_paths):
        label_str = r["label"] if r["label"] != "UNKNOWN" else "—"
        print(f"  {r['crop_id']:>4}  [{label_str:^6}]  {r['score']:>8.4f}  {img_path.name}")

    defects = [r for r in results if r["label"] == "DEFECT"]

    print(f"\n{'═'*57}")
    print(f"  ÖZET")
    print(f"{'─'*57}")
    print(f"  Toplam görüntü     : {len(results)}")
    print(f"  Toplam süre        : {total_elapsed:.2f} sn")
    print(f"  Ortalama/görüntü   : {total_elapsed/len(results)*1000:.0f} ms")
    print(f"  DEFECT : {len(defects)}  |  NORMAL : {len(results)-len(defects)}")
    print(f"{'═'*57}")

    summary_path = save_dir / "summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("id,dosya,label,score\n")
        for r, img_path in zip(results, image_paths):
            f.write(f"{r['crop_id']},{img_path.name},{r['label']},{r['score']:.4f}\n")

    print(f"\n  Özet : {summary_path}")


if __name__ == "__main__":
    main()