import numpy as np
import copy

def soft(x, sigma):
    z = np.sign(x) * np.maximum(np.abs(x)-sigma, 0)
    return z

def MM(A, y, x_k, lamb=0.1, c=1e5):
    new_x = soft(1/c*(np.matmul(A.T, y-A.dot(x_k)))+x_k, lamb/c)
    return new_x

def Gradient(A, y, x_k, gamma=1e-5, lamb = 0.1):
    grad = 2 * np.matmul(A.T, (np.matmul(A, x_k) - y)) + 2 * lamb * x_k
    return grad 

if __name__ == "__main__":
    data = np.loadtxt('./data_problem1/data.txt')
    m, n = data.shape
    label = np.loadtxt('./data_problem1/label.txt')
    label = np.reshape(label,[m,1])
    gamma = 1e-5

    x = np.random.random([n,1])
    dist = 1
    while dist > 1e-3:
        old_x = copy.deepcopy(x)
        old_dist = np.matmul(data, old_x) - label
        x = MM(data, label, x)
        new_dist = np.matmul(data, x) - label
        dist = np.linalg.norm(new_dist - old_dist)
    x = x.reshape([n,1])
    np.savetxt('sol1.txt', x, fmt="%.6f")
    # Comparison
    print(np.linalg.norm(x.reshape(-1,), ord=0))
    loss = np.linalg.norm(np.matmul(data, x) - label) ** 2 
    print(loss)

    x = np.random.random([n,1])
    grad = 1
    while np.linalg.norm(grad) > 1e-3:
        grad = Gradient(data, label, x)
        x = x - gamma * grad
    x = x.reshape([n,1])
    np.savetxt('sol2.txt', x, fmt="%.6f")
    # Comparison
    print(np.linalg.norm(x.reshape(-1,), ord=0))
    loss = np.linalg.norm(np.matmul(data, x) - label) ** 2 
    print(loss)

