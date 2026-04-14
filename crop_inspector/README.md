# CropInspector

Görüntüyü crop'lara böler, her crop'u `crop_id % 10` kuralıyla
10 GaussianModel'den birine yönlendirir. Backbone (ResNet18) yalnızca
**bir kez** yüklenir ve tüm modeller tarafından paylaşılır.

## Proje yapısı

```
crop_inspector/
├── inspector.py               ← CropInspector (1 backbone + 10 Gaussian)
├── model/
│   ├── feature_extractor.py   ← Paylaşılan backbone (ResNet18 early layers)
│   └── gaussian_model.py      ← Bağımsız anomali modeli
├── core/
│   └── transforms.py          ← Görüntü ön işleme
├── utils/
│   └── cropper.py             ← Görüntüyü crop'lara böler
├── scripts/
│   ├── train.py               ← Tek model eğitimi (--model_idx ile)
│   └── run.py                 ← Tam akış: crop → analiz → sonuç
└── weights/
    ├── model_0/gaussian_model.npz
    ├── model_1/gaussian_model.npz
    └── ...
```

## Mimari

```
Görüntü
  ↓
FeatureExtractor  ← 1× paylaşılan backbone (~11 MB, bir kez yüklenir)
  ↓ (1, 192, H/4, W/4)
  ├─→ GaussianModel[0]   weights/model_0/gaussian_model.npz
  ├─→ GaussianModel[1]   weights/model_1/gaussian_model.npz
  │   ...
  └─→ GaussianModel[9]   weights/model_9/gaussian_model.npz
        ↓
    Mahalanobis mesafesi → heatmap + skor
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Her modeli ayrı ayrı eğit

```bash
python scripts/train.py \
    --model_idx   0 \
    --data_dir    data/train/model_0/normal \
    --weights_dir weights/

python scripts/train.py \
    --model_idx   1 \
    --data_dir    data/train/model_1/normal \
    --weights_dir weights/

# ... 9'a kadar
```

Çıktı: `weights/model_<idx>/gaussian_model.npz`

### 2. Görüntüyü analiz et

```bash
python scripts/run.py \
    --image     foto.png \
    --weights   weights/ \
    --crop_size 256 256 \
    --threshold 4.0 \
    --save_dir  results/
```

## Yönlendirme kuralı

| crop_id | crop_id % 10 | GaussianModel |
|---------|-------------|---------------|
| 0       | 0           | model_0       |
| 1       | 1           | model_1       |
| 10      | 0           | model_0       |
| 11      | 1           | model_1       |
| 99      | 9           | model_9       |

## Python API

```python
from inspector     import CropInspector
from utils.cropper import crop_image

# Yükle (backbone 1×, Gaussian 10×)
inspector = CropInspector(weights_dir="weights/", device="cpu")

# Görüntüyü crop'la
crops = crop_image("foto.png", crop_size=(256, 256), save_dir="crops/")

# Analiz et
for c in crops:
    result = inspector.inspect(c["crop_id"], c["path"], threshold=4.0)
    print(result)
    # {"crop_id": 7, "model_idx": 7, "score": 2.34, "label": "NORMAL", "heatmap": ...}
```
