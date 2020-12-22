import numpy as np
import txt_lw as tlw
import math
import copy

mlambda = 0.1

'''
the gradient of loss function   2A^{\top}(Ax-y) + 2*lambda x
'''
def gradient(y, A, x):
    part1 = 2 * np.transpose(A)
    part2 = np.matmul(A, x) - y
    return np.matmul(part1, part2) + 2*mlambda*x

def loss(A, x, y):
    """
    the loss function
    """
    return np.linalg.norm(np.matmul(A, x) - y.reshape(-1,1))

'''
the gradient decent method
'''
def gd(y, A, x):
    ITERATION_TIMES = 10000
    step_size = 1e-5
    for iterTime in range(ITERATION_TIMES):
        #loss = loss_func(y, A, x)
        grad = gradient(y, A, x)
        if np.linalg.norm(grad) < 1e-2:
            print("Iteration finishes: {} iterations".format(iterTime))
            break
        x = x - step_size * grad
    print("loss is {}".format(loss(A,x,y))) 
    print(np.linalg.norm(grad))   
    return x



A = tlw.data_loader()
y = tlw.label_loader()
y = y.reshape(-1,1)
init_x = 20 * np.random.random_sample((A.shape[1], 1 )) - 10
init_x = init_x.reshape(-1,1)

final_x = gd(y, A, init_x)
tlw.sol_save("sol2.txt", final_x)
