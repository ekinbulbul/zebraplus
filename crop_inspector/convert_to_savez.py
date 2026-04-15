"""
savez_compressed → savez dönüşümü.
Decompress overhead ortadan kalkar (~2400ms → ~5ms).
Dosya boyutu 2x büyür ama yükleme çok hızlanır.
"""
import numpy as np
import glob

files = glob.glob('weights/model_*/gaussian_model.npz')
print(f'{len(files)} dosya bulundu.')

for f in files:
    d = np.load(f)
    np.savez(
        f.replace('.npz', ''),
        mean=np.array(d['mean']),
        precision=np.array(d['precision']),
        spatial=d['spatial'],
        fp16=d['fp16'],
    )
    print(f'  OK: {f}')

print('Tamamlandi.')
