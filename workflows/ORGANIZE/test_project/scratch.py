# scratch — quick test of noise term formula
# NOT a real module; do not import

import numpy as np

arr = np.random.normal(0, 1, (4, 4))
print(arr.mean(), arr.std())
