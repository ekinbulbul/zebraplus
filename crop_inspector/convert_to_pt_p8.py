"""
weights_p8/ altındaki tüm npz dosyalarını pt'ye dönüştürür.
"""
import torch
import numpy as np
import glob
from pathlib import Path

files = glob.glob('weights_p8/model_*/gaussian_model.npz')
print(f'{len(files)} dosya bulundu.\n')

for f in files:
    npz_path = Path(f)
    pt_path  = npz_path.with_suffix('.pt')
    d = np.load(str(npz_path))
    torch.save({
        'mean'     : torch.from_numpy(np.array(d['mean'])),
        'precision': torch.from_numpy(np.array(d['precision'])),
        'spatial'  : tuple(d['spatial'].tolist()),
        'fp16'     : bool(d['fp16']),
    }, str(pt_path))
    print(f'  OK: {pt_path}')

print(f'\nTamamlandı.')
