import numpy as np

a = np.random.random_integers(0, 10, (4, 4, 4))
print(a)
print(np.max(a, axis=2))