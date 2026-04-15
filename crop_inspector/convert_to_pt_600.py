"""
weights_pool8_600/model_0 npz → pt dönüştürür.
Sonra kopyalama yapılacak.
"""
import torch
import numpy as np
from pathlib import Path

npz_path = Path('weights_pool8_600/model_0/gaussian_model.npz')
pt_path  = npz_path.with_suffix('.pt')

d = np.load(str(npz_path))
torch.save({
    'mean'     : torch.from_numpy(np.array(d['mean'])),
    'precision': torch.from_numpy(np.array(d['precision'])),
    'spatial'  : tuple(d['spatial'].tolist()),
    'fp16'     : bool(d['fp16']),
}, str(pt_path))

print(f'OK: {pt_path}  ({pt_path.stat().st_size/1e6:.0f} MB)')
print('Şimdi kopyalayabilirsin.')
