import time, sys, glob
sys.path.insert(0, '.')
from inspector import CropInspector

inspector = CropInspector(weights_dir='weights/', img_size=(1200, 1200))
imgs = sorted(glob.glob('photos/*.png') + glob.glob('photos/*.bmp') + glob.glob('photos/*.jpg'))[:5]

print(f"\n{len(imgs)} görüntü test ediliyor...\n")
for i, img in enumerate(imgs):
    t0 = time.perf_counter()
    r  = inspector.inspect(i, img, threshold=4.0)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  crop {i}: {ms:.0f} ms  score={r['score']}  label={r['label']}")
