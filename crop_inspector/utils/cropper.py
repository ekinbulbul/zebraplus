"""
utils/cropper.py
-----------------
Bir görüntüyü sabit boyutlu crop'lara böler.
Her crop'a sıralı bir ID verir (0, 1, 2, ...).

Kullanım:
    crops = crop_image("photo.png", crop_size=(256, 256), save_dir="crops/")
    # → [{"crop_id": 0, "path": "crops/crop_0000.png", "x": 0, "y": 0}, ...]
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def crop_image(
    image_path: str,
    crop_size: tuple[int, int] = (256, 256),
    save_dir: str = "crops/",
    overlap: int = 0,
) -> list[dict]:
    """
    Görüntüyü crop_size boyutunda parçalara böler.
    Kenar kısımları görüntü sınırında kesilir (padding yok).

    Args:
        image_path : kaynak görüntü
        crop_size  : (H, W) her crop'un boyutu
        save_dir   : crop'ların kaydedileceği klasör
        overlap    : piksel cinsinden örtüşme miktarı (default 0)

    Returns:
        [{"crop_id": int, "path": str, "x": int, "y": int, "w": int, "h": int}, ...]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Görüntü açılamadı: {image_path}")

    img_h, img_w = img.shape[:2]
    crop_h, crop_w = crop_size
    step_h = crop_h - overlap
    step_w = crop_w - overlap

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    crops = []
    crop_id = 0

    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            # Sınırı aşma
            x_end = min(x + crop_w, img_w)
            y_end = min(y + crop_h, img_h)
            patch = img[y:y_end, x:x_end]

            fname = Path(save_dir) / f"crop_{crop_id:04d}.png"
            cv2.imwrite(str(fname), patch)

            crops.append({
                "crop_id" : crop_id,
                "path"    : str(fname),
                "x"       : x,
                "y"       : y,
                "w"       : x_end - x,
                "h"       : y_end - y,
            })

            crop_id += 1
            x += step_w
        y += step_h

    return crops
