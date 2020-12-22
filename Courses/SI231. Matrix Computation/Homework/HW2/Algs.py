import random
import numpy as np
import copy
import time
from tqdm import tqdm

def Inv(M,n):
    M = copy.deepcopy(M)
    I = np.zeros((n,n))
    for i in range(n):
        I[i][i] = 1
    ## Make diagonal elements as non zero
    for i in range(n):
        if(M[i][i]==0):
            for j in range(n):
                if(M[j][i]!=0):
                    M[i,] = M[i,] + M[j,]
                    I[i,] = I[i,] + I[j,]
                    break
    ## perform Gauss Jordan elimination
    for i in range(n):
        for j in range(n):
            if(j!=i):
                I[j,] = I[j,] - M[j][i]*I[i,]/M[i][i]
                M[j,] = M[j,] - M[j][i]*M[i,]/M[i][i]
    ## Normalize the M[i][i] values of Matrix M
    for i in range(n):
        I[i,] = I[i,]/M[i][i]
        M[i,] = M[i,]/M[i][i]
    ## I is now the inverted matrix of M
    return I


# def det(matrix):
#     # Base case for matrix of shape (2,2)
#     if matrix.shape[0] == 2:
#         return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]

#     # Implement recursion function.
#     determinant = 0
#     for i in range(matrix.shape[0]):
#         temp_matrix = matrix[1:,:].copy()
#         temp_matrix = np.delete(temp_matrix, i, 1)
#         determinant = determinant + ((-1)**i)*matrix[0][i]*det(temp_matrix)
#     return determinant


def cramer(d,b):
    d = copy.deepcopy(d)
    b = copy.deepcopy(b)
    d_i = []
    for i in range(b.shape[0]):
        d_i.append(copy.deepcopy(d))
        d_i[i][:,i] = b
        pass
    x = []
    for i in range(b.shape[0]):
        x.append(np.linalg.det(d_i[i]) /np.linalg.det(d) + 0.0001)
    return x


def Gauss(A):
    n = A.shape[0]
    B = copy.deepcopy(A)
    t = np.zeros(n)
    x = np.zeros(n)
    #print(A)
    # Forward
    for i in range(n):
        t[i+1:] = B[i+1:, i]/ B[i, i]
        tmp1 = t[i+1:].reshape(-1, 1)
        tmp2 = B[i, i+1:].reshape(1, -1)
        B[i+1:, i+1:] -= np.matmul(tmp1, tmp2)
        B[i+1:, i] = np.zeros(len(B[i+1:, i]))
        B[i, i:] /= B[i][i]
    # Backwarkd
    for i in range(n-1):
        r = n - 1 - i
        t[0: r] = B[0:r, r] / B[r, r]
        tmp1 = t[0:r].reshape(-1,1)
        tmp2 = B[r].reshape(1,-1)
        #print(tmp1,tmp2)
        B[0:r] -= np.matmul(tmp1, tmp2)


def LU(M):
    n = len(M)
    L = np.eye(n)
    U = copy.deepcopy(M)
    t = np.zeros(n)
    for k in range(n-1):
        t[k+1:] = U[k+1:, k]/U[k, k]
        tmp1 = t[k+1:].reshape(-1, 1)
        tmp2 = U[k, k+1:].reshape(1, -1)
        U[k+1:, k+1:] = U[k+1:, k+1:] - np.matmul(tmp1, tmp2)
        U[k+1:, k] = np.zeros(len(U[k+1:, k]))
        L[k+1:, k] = t[k+1:]
    return L,U

def LUsolve(A,b,n):
    L, U = LU(A)
    # Solve Lz = b
    n = len(A)
    y = np.zeros((n, 1))
    b = np.array(b).reshape(n,1) 
    for i in range(len(A)):
        t = 0
        for j in range(i):
            t += L[i][j]* y[j][0]
        y[i][0] = b[i][0] - t
    # Solve Ux = b
    X = np.zeros((n, 1))
    for i in range(len(A)-1,-1,-1):
        t = 0
        for j in range(i+1,len(A)):
            t += U[i][j]*X[j][0]
        t = y[i][0] - t
        if t != 0 and U[i][i] == 0:
            return 0
        X[i] = t/U[i][i]

    return X



if __name__ == "__main__":
    ns = np.linspace(100,1000,19)
    inv_time = np.zeros(19)
    cramer_time = np.zeros(19)
    gauss_time = np.zeros(19)
    lu_time = np.zeros(19)
    for i, n in tqdm(enumerate(ns)):
        np.random.seed(5)
        mat = np.random.random((int(n),int(n))) * 10 + 12
        mat_list = list(mat)
        b = np.random.random(int(n)) * 10 + 12
        mat_aug = np.concatenate((mat,np.reshape(b.T,(int(n),1))),axis=1)


        stime = time.time()
        solution = LUsolve(mat, b, n)
        etime = time.time()
        lu_time[i] = etime - stime

        stime = time.time()
        solution = Inv(mat, int(n)).dot(b)
        etime = time.time()
        inv_time[i] = etime - stime

        stime = time.time()
        solution = cramer(mat, b)
        etime = time.time()
        cramer_time[i] = etime - stime

        stime = time.time()
        solution = Gauss(mat_aug)
        #print(solution)
        etime = time.time()
        gauss_time[i] = etime - stime

        np.save('inv_time.npy', inv_time)
        np.save('cramer_time.npy', cramer_time)
        np.save('gauss_time.npy', gauss_time)
        np.save('lu_time.npy', lu_time)
