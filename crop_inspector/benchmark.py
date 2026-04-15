"""
benchmark.py — gaussian.to(cpu) olmadan ölçüm
"""
import argparse, time, sys, torch, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",   type=str, default="weights/")
    p.add_argument("--model_idx", type=int, default=0)
    p.add_argument("--repeat",    type=int, default=5)
    return p.parse_args()

def measure(label, fn, repeat=5):
    fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    avg, mn, mx = sum(times)/len(times), min(times), max(times)
    print(f"  {label:40} : ort={avg:7.1f}ms  min={mn:7.1f}ms  maks={mx:7.1f}ms")

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path   = f"{args.weights}/model_{args.model_idx}/gaussian_model"

    print(f"Device : {device}  |  Tekrar : {args.repeat}\n")

    from model.gaussian_model    import GaussianModel
    from model.feature_extractor import FeatureExtractor
    from core.transforms         import build_transform, load_batch
    import torch.nn.functional as F

    backbone  = FeatureExtractor(pretrained=True, freeze=True).to(device)
    backbone.eval()
    transform = build_transform((1200, 1200))
    imgs = glob.glob("photos/*.png") + glob.glob("photos/*.jpg") + glob.glob("photos/*.bmp")

    # 1. Disk → RAM
    measure("Disk → RAM (load)",
            lambda: GaussianModel(device="cpu").load(path) or None,
            args.repeat)

    # 2. RAM → GPU (to cpu yok)
    def load_to_gpu():
        g = GaussianModel(device="cpu")
        g.load(path)
        g.to(device)
        return g

    measure("Disk → RAM → GPU", load_to_gpu, args.repeat)

    # 3. Del (to cpu yok)
    def load_use_del():
        g = GaussianModel(device="cpu")
        g.load(path)
        g.to(device)
        del g
        torch.cuda.empty_cache()

    measure("Load → GPU → del (to_cpu yok)", load_use_del, args.repeat)

    # 4. Tam crop (gerçek senaryo)
    print()
    if imgs:
        dummy = load_batch([imgs[0]], transform).to(device)
        def full_crop():
            g = GaussianModel(device="cpu")
            g.load(path)
            g.to(device)
            with torch.no_grad():
                f = backbone(dummy)
                f = F.avg_pool2d(f, 4, 4)
                g.mahalanobis_map(f)
            del g
            torch.cuda.empty_cache()

        measure("Tam crop (load+gpu+mahal+del)", full_crop, args.repeat)

        # 75 crop tahmini
        times = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            full_crop()
            times.append((time.perf_counter()-t0)*1000)
        avg = sum(times)/len(times)
        print(f"\n  Ortalama : {avg:.0f} ms/crop")
        print(f"  75 crop  : {avg*75/1000:.1f} sn")
        print(f"  200 crop : {avg*200/1000:.1f} sn")
    else:
        print("  photos/ boş, tam crop atlandı.")

if __name__ == "__main__":
    main()