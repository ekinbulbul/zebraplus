"""
scripts/run.py
---------------
Bir klasördeki tüm görüntüleri alır, her birini crop'lara böler
ve her crop'u ilgili model instance (crop_id % 10) ile analiz eder.

Crop ID'leri görüntüler arası sıfırlanmaz — tüm klasör boyunca
artar (görüntü 0'ın son crop'u neredeyse oradan devam eder).
Böylece her crop'un global olarak benzersiz bir ID'si olur.

Kullanım:
    python scripts/run.py \
        --image_dir  photos/ \
        --weights    weights/ \
        --crop_size  256 256 \
        --threshold  4.0 \
        --save_dir   results/

Tek görüntü için:
    python scripts/run.py \
        --image_dir  photos/ \        ← tek dosya içeren klasör
        --weights    weights/

Çıktı yapısı:
    results/
        img_001/
            crops/          ← ham crop'lar
            crop_0000_result.png
            crop_0001_result.png
            ...
        img_002/
            ...
        summary.txt         ← tüm klasörün özeti
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspector     import CropInspector
from utils.cropper import crop_image
import cv2
import numpy as np

EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]


def parse_args():
    p = argparse.ArgumentParser(description="CropInspector — Klasör analizi")
    p.add_argument("--image_dir",  type=str, required=True,
                   help="Analiz edilecek görüntülerin bulunduğu klasör")
    p.add_argument("--weights",    type=str, default="weights/")
    p.add_argument("--crop_size",  type=int, nargs=2, default=[256, 256])
    p.add_argument("--overlap",    type=int, default=0)
    p.add_argument("--threshold",  type=float, default=None)
    p.add_argument("--save_dir",   type=str, default="results/")
    p.add_argument("--device",     type=str, default=None)
    p.add_argument("--img_size",   type=int, nargs=2, default=[256, 256])
    p.add_argument("--half",       action="store_true", default=False)
    return p.parse_args()


def collect_images(image_dir: str) -> list[Path]:
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


def save_heatmap_overlay(result: dict, crop_path: str, save_dir: str):
    img = cv2.imread(crop_path)
    if img is None:
        return
    h, w = img.shape[:2]
    hm = cv2.resize(result["heatmap"], (w, h))
    hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay  = cv2.addWeighted(img, 0.6, hm_color, 0.4, 0)
    color = (0, 0, 220) if result["label"] == "DEFECT" else (0, 180, 0)
    cv2.putText(overlay,
                f"ID:{result['crop_id']} M:{result['model_idx']} {result['label']}",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    out = Path(save_dir) / f"{Path(crop_path).stem}_result.png"
    cv2.imwrite(str(out), overlay)


def print_image_summary(image_name: str, results: list[dict], threshold):
    defects = [r for r in results if r["label"] == "DEFECT"]
    status  = "DEFECT" if defects else "NORMAL"
    print(f"  [{status}]  {image_name:30s}  "
          f"crop:{len(results)}  defect:{len(defects)}  "
          f"max_score:{max(r['score'] for r in results):.4f}")


def write_summary(summary_path: Path, all_results: dict, threshold):
    lines = ["görüntü, toplam_crop, defect_crop, max_score, durum\n"]
    for img_name, results in all_results.items():
        defects   = [r for r in results if r["label"] == "DEFECT"]
        max_score = max(r["score"] for r in results)
        durum     = "DEFECT" if defects else ("NORMAL" if threshold else "UNKNOWN")
        lines.append(f"{img_name},{len(results)},{len(defects)},{max_score:.4f},{durum}\n")
    summary_path.write_text("".join(lines), encoding="utf-8")


def main():
    args = parse_args()

    image_paths = collect_images(args.image_dir)
    save_dir    = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {len(image_paths)} görüntü bulundu → {args.image_dir}")

    # Inspector bir kez yüklenir — tüm görüntüler için paylaşılır
    inspector = CropInspector(
        weights_dir=args.weights,
        img_size=tuple(args.img_size),
        device=args.device,
        half=args.half,
    )

    print(f"\n[run] Analiz başlıyor...\n")
    print(f"  {'durum':8}  {'dosya':30}  {'crop':>5}  {'defect':>6}  {'max_score':>9}")
    print("  " + "─" * 65)

    global_crop_id = 0   # klasör boyunca artan global ID
    all_results    = {}

    for img_path in image_paths:
        img_name  = img_path.stem
        img_dir   = save_dir / img_name
        crops_dir = img_dir / "crops"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Crop'la — global ID'den devam et
        raw_crops = crop_image(
            image_path=str(img_path),
            crop_size=tuple(args.crop_size),
            save_dir=str(crops_dir),
            overlap=args.overlap,
        )

        # Yerel crop ID'lerini global ID'ye çevir
        crop_pairs = []
        crop_path_map = {}
        for c in raw_crops:
            gid = global_crop_id + c["crop_id"]
            crop_pairs.append((gid, c["path"]))
            crop_path_map[gid] = c["path"]
        global_crop_id += len(raw_crops)

        # Analiz
        results = inspector.inspect_batch(crop_pairs, threshold=args.threshold)
        all_results[img_name] = results

        # Overlay kaydet
        for r in results:
            save_heatmap_overlay(r, crop_path_map[r["crop_id"]], str(img_dir))

        print_image_summary(img_name, results, args.threshold)

    # Genel özet dosyası
    summary_path = save_dir / "summary.csv"
    write_summary(summary_path, all_results, args.threshold)

    total_crops   = sum(len(v) for v in all_results.values())
    total_defects = sum(
        len([r for r in v if r["label"] == "DEFECT"])
        for v in all_results.values()
    )

    print(f"\n{'─'*67}")
    print(f"  Toplam görüntü : {len(image_paths)}")
    print(f"  Toplam crop    : {total_crops}")
    print(f"  Defect crop    : {total_defects}")
    print(f"  Özet           : {summary_path}")
    print(f"  Sonuçlar       : {save_dir}")


if __name__ == "__main__":
    main()
