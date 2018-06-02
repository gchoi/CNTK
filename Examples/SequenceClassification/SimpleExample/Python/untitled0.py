import numpy as np

I = [[8, 6, 2, 7], [6, 2, 4, 1], [5, 8, 5, 2], [3, 0, 3, 2]]
K = [[4, 3], [7, 2]]

I = np.asarray(I)
K = np.asarray(K)

subI = I[2:4,2:4]

print(np.sum(np.multiply(subI,K)))