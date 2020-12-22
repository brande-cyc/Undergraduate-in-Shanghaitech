import numpy as np
import txt_lw as tlw
import math
import copy

c = 1e+5
mlambda = 0.1
MAX_ITER = 10000


def soft(x, delta):
    """
    the soft-thresholding operator
    """
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        z[i] = np.sign(x[i]) * max(abs(x[i])-delta, 0)
    return z

def loss(A, x, y):
    """
    the loss function
    """
    return np.linalg.norm(np.matmul(A, x) - y.reshape(-1,1))

def MaMi(A_hat, y_hat):
    """
    the Majorization-Minimization method
    """
    A = copy.deepcopy(A_hat)
    y = copy.deepcopy(y_hat)
    m, n = A.shape
    x = 20 * np.random.random_sample((n, 1)) - 10
    
    for iterTime in range(MAX_ITER):
        currLoss = loss(A, x, y)
        if currLoss < 54.275:
            print("Iteration finishes: {} iterations".format(iterTime))
            break
        firstItem = 1.0/c * np.matmul(np.transpose(A), y.reshape(-1,1)-np.matmul(A, x)) + x
        secondItem = mlambda/c
        x = soft(firstItem, secondItem)
    print("loss is {}".format(loss(A,x,y))) 
    return x


A = tlw.data_loader()
y = tlw.label_loader()
x = MaMi(A, y)

tlw.sol_save("sol1.txt", x)