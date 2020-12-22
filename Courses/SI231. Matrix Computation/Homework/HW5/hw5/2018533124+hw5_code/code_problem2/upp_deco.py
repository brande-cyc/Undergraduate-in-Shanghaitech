import numpy as np
import math
import copy
import matplotlib.pyplot as plt

from svd_deco import SVD_deco
algo_1_res = []
algo_deco_res = []

mList = np.linspace(1,30,30, dtype=int)
for m in mList:
    A = np.ones((m,m))
    for i in range(m):
        for j in range(m):
            if i == j:
                A[i,j] = 0.1
            elif i > j:
                A[i,j] = 0
    _, Sigma_1, _ = SVD_deco(A)
    Sigma_1_list = np.diag(Sigma_1)
    sigma_1_min = Sigma_1_list[-1]
    algo_1_res.append(np.log10(sigma_1_min))

    _, Sigma_2_list, _ = np.linalg.svd(A)
    Sigma_2_list = sorted(Sigma_2_list, reverse=True)
    sigma_2_min = Sigma_2_list[-1]
    algo_deco_res.append(np.log10(sigma_2_min))

plt.title("Smallest singular value")
plt.xlabel('matrix size / m')
plt.ylabel('value in log scale')

plt.plot(mList, algo_1_res, label = "algo 1")
plt.plot(mList, algo_deco_res, label = "default svd")

plt.legend()
plt.show()




