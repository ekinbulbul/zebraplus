"""
Tüm gaussian_model.npz dosyalarını gaussian_model.pt formatına dönüştürür.
Sonra npz dosyalarını siler.
"""
import torch
import numpy as np
import glob
from pathlib import Path

files = glob.glob('weights/model_*/gaussian_model.npz')
print(f'{len(files)} dosya bulundu.\n')

for f in files:
    npz_path = Path(f)
    pt_path  = npz_path.with_suffix('.pt')

    # Yükle
    d = np.load(str(npz_path))
    mean      = torch.from_numpy(np.array(d['mean']))
    precision = torch.from_numpy(np.array(d['precision']))
    spatial   = tuple(d['spatial'].tolist())

    # PT olarak kaydet
    torch.save({
        'mean'     : mean,
        'precision': precision,
        'spatial'  : spatial,
        'fp16'     : bool(d['fp16']),
    }, str(pt_path))

    print(f'  OK: {pt_path}  ({pt_path.stat().st_size/1e6:.0f} MB)')

print(f'\nTamamlandı. {len(files)} dosya dönüştürüldü.')
print('npz dosyalarını silmek ister misin? (elle sil ya da aşağıdaki kodu aç)')
# Silmek için:
# for f in files:
#     Path(f).unlink()
