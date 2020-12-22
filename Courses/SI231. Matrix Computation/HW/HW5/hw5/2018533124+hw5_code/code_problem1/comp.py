import numpy as np
import txt_lw as tlw
import math
import copy

def loss(A, x, y):
    """
    the loss function
    """
    return np.linalg.norm(np.matmul(A, x) - y.reshape(-1,1))

A = tlw.data_loader()
y = tlw.label_loader()
y = y.reshape(-1,1)
x_mm = np.array(np.loadtxt('sol1.txt'))
x_mm = x_mm.reshape(-1,1)
x_gd = np.array(np.loadtxt('sol2.txt'))
x_gd = x_gd.reshape(-1,1)

loss_mm = loss(A, x_mm, y)**2
loss_gd = loss(A, x_gd, y)**2

l0norm_mm = np.linalg.norm(x_mm.reshape(-1,), ord=0)
l0norm_gd = np.linalg.norm(x_gd.reshape(-1,), ord=0)

print("Majorization-Minimization l0 norm: {} ".format(l0norm_mm))
print("gradient descent l0 norm: {} ".format(l0norm_gd))
print("Majorization-Minimization loss: {} ".format(loss_mm))
print("gradient descent loss: {} ".format(loss_gd))



