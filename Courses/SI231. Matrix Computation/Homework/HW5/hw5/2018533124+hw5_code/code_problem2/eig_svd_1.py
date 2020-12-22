import numpy as np
import math
import copy

A = np.array([[0,1,0,0],[0,0,2,0],[0,0,0,3],[1.0/6000.0,0,0,0]])

eigs = np.linalg.eigvals(A)
print("eigenvalues: \n{}".format(eigs))

print("=========================================================")
print("=========================================================")

U, sins, Vh = np.linalg.svd(A)
print("singular values: \n{}".format(sins))

