import time
import numpy as np

d = np.load('weights/model_0/gaussian_model.npz')

t0 = time.perf_counter()
p = np.array(d['precision'])
t1 = time.perf_counter()

print(f'np.array(precision): {(t1-t0)*1000:.1f} ms')
print(f'dtype: {p.dtype}  size: {p.nbytes/1e6:.0f} MB')
