import numpy as np
import math
import copy

def SVD_deco(A_hat, modifi = True):
    A = copy.deepcopy(A_hat)
    A = A.astype(np.float64)
    m,n = A.shape
    if m<n:
        A = np.transpose(A)
    At = np.transpose(A)
    #=========================================
    # step 1
    ATA = np.matmul(At, A)

    #=========================================
    # step 2
    eigs, V = np.linalg.eigh(ATA)

    eigs = sorted(eigs, reverse=True)
    #=========================================
    # step 3
    Sigma = np.zeros(A.shape)
    for i in range(len(eigs)):
        Sigma[i,i] = np.sqrt(eigs[i])

    #=========================================
    # step 4
    q,r = np.linalg.qr(np.matmul(A, V),mode='complete')
    U = q
    if not modifi:
        return U, Sigma, V
    r_diag = np.zeros(r.shape[1])
    for i in range(len(r_diag)):
        r_diag[i] = r[i,i]
        
        # conver negative diag entry to positive
        negFlag = False
        if r_diag[i] < 0:
            r_diag[i] = -r_diag[i]
            negFlag = True

        # sort in descending order
        for j in range(i, 0, -1):
            if r_diag[j] > r_diag[j-1]:
                temp = copy.deepcopy(r_diag[j])
                r_diag[j] = copy.deepcopy(r_diag[j-1])
                r_diag[j-1] = temp

                tempList = copy.deepcopy(U[:,j])
                U[:,j] = copy.deepcopy(U[:,j-1])
                U[:,j-1] = tempList

                tempList = copy.deepcopy(V[j,:])
                V[j,:] = copy.deepcopy(V[j-1,:])
                V[j-1,:] = tempList
        if negFlag == True:
            U[:,i] = np.negative(U[:,i])
            V = np.negative(V)
    return U, Sigma, V

if __name__ == "__main__":
    B = np.array([[3,2],[1,4],[0.5, 0.5]],dtype=np.float64)
    U, Sigma, V = SVD_deco(B)
    print(U)
    print(Sigma)
    print(V)
    print(np.matmul(U, np.matmul(Sigma, V)))



