# tmp — one-off experiment to test noise scaling
# Created 2026-03-10; results pasted into lab notebook; can delete.

import numpy as np

for sigma in [0.1, 0.5, 1.0, 2.0]:
    samples = np.random.normal(0, sigma, 10_000)
    print(f"sigma={sigma}: mean={samples.mean():.4f} std={samples.std():.4f}")
