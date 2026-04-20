import numpy as np
x = 0
for t in range(1000000):
    x += np.sin(t)
print(x)
