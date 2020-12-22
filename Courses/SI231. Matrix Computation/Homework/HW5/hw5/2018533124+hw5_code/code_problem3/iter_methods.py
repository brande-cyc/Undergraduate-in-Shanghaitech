import numpy as np
import math
import copy
import matplotlib.pyplot as plt

JAC = "Jacobi"
GS = "Gauss-Seidel"
SOR = "SOR"

METHODerror_list = [[],[],[]]

def Jacobi_iter(A, x, b):
    """
    Jacobi iteration
    """
    n = A.shape[1]
    x_new = copy.deepcopy(x)
    for i in range(n):
        x_new[i] = b[i]
        for j in range(n):
            if j != i:
                x_new[i] -= A[i,j]*x[j]
        x_new[i] /= A[i,i]
    return x_new

def GaussSeidel_iter(A, x, b):
    """
    Gauss-Seidel iteration
    """
    n = A.shape[0]
    x_new = copy.deepcopy(x)
    for i in range(n):
        x_new[i] = b[i]
        for j in range(n):
            if j > i:
                x_new[i] -= A[i,j]*x_new[j]
            if j < i:
                x_new[i] -= A[i,j]*x[j]
        x_new[i] /= A[i,i]
    return x_new

def SOR_iter(A, x, b, OMEGA):
    """
    SOR iteration
    """
    n = A.shape[0]
    x_new = copy.deepcopy(x)
    for i in range(n):
        x_new[i] = b[i]
        for j in range(n):
            if j > i:
                x_new[i] -= A[i,j]*x_new[j]
            if j < i:
                x_new[i] -= A[i,j]*x[j]
        x_new[i] /= A[i,i]
        x_new[i] = x_new[i] * OMEGA
        x_new[i] = x_new[i] + (1-OMEGA) * x[i]
    return x_new

def iter_process(A, x, b, MAX_ITER, bound, METHOD, OMEGA, x_GroundTruth):
    x_prev = copy.deepcopy(x)
    x_curr = copy.deepcopy(x)
    for k in range(MAX_ITER+1):
        if METHOD == JAC:
            x_curr = Jacobi_iter(A,x_prev,b)
            METHODerror_list[0].append(np.log10(np.linalg.norm(x_GroundTruth - x_curr)))
        elif METHOD == GS:
            x_curr = GaussSeidel_iter(A,x_prev,b)
            METHODerror_list[1].append(np.log10(np.linalg.norm(x_GroundTruth - x_curr)))
        elif METHOD == SOR:
            x_curr = SOR_iter(A, x_prev, b, OMEGA)
            METHODerror_list[2].append(np.log10(np.linalg.norm(x_GroundTruth - x_curr)))
        #print(np.linalg.norm(x_curr-x_prev))

        if np.linalg.norm(x_curr-x_prev) < bound:
            print(k)
            print(np.linalg.norm(x_curr-x_prev))
            return x_curr,k
        x_prev = x_curr
    return x_curr, 200

if __name__ == "__main__":
    indexList = [1,2]
    OMEGA_list = np.linspace(0.8,1.0,10,endpoint=True)
    METHOD_list = [JAC, GS, SOR]
    SECOND_PLOT = True #plot problem2 if True, otherwise plot problem3

    if SECOND_PLOT == True:
        
        iterTime_list = [[],[]]
        for ind in indexList:
            A = np.array(np.loadtxt('../data_problem3/A{}.txt'.format(ind),dtype=np.float64))
            b = np.array(np.loadtxt('../data_problem3/b{}.txt'.format(ind),dtype=np.float64))
            x_GroundTruth = np.array(np.loadtxt('../data_problem3/x{}.txt'.format(ind),dtype=np.float64))
            x_init = np.random.random_sample((A.shape[1]))
            for OMEGA in OMEGA_list:
                print(OMEGA)
                x_res, iter_time = iter_process(A, x_init, b, 200, 1e-15, SOR, OMEGA, x_GroundTruth)
                iterTime_list[ind-1].append(iter_time)
        plt.title("# of iterations  v.s.  omega")
        plt.xlabel('omega')
        plt.ylabel('# of iterations')

        plt.plot(OMEGA_list, iterTime_list[0], label = "size = 10")
        plt.plot(OMEGA_list, iterTime_list[1], label = "size = 1000")

        plt.legend()
        plt.show()

    if SECOND_PLOT == False:
        
        for ind in indexList:
            METHODerror_list = [[],[],[]]
            MAXITER_list = []
            A = np.array(np.loadtxt('../data_problem3/A{}.txt'.format(ind),dtype=np.float64))
            b = np.array(np.loadtxt('../data_problem3/b{}.txt'.format(ind),dtype=np.float64))
            x_GroundTruth = np.array(np.loadtxt('../data_problem3/x{}.txt'.format(ind),dtype=np.float64))
            x_init = np.random.random_sample((A.shape[1]))
            for METHOD in METHOD_list:
                x_res, iter_time = iter_process(A, x_init, b, 200, 1e-15, METHOD, 1.0, x_GroundTruth)
                MAXITER_list.append(iter_time)
                
            plt.title("error, size = {}".format(A.shape[1]))
            plt.xlabel('k')
            plt.ylabel('error in log')

            plt.plot(np.linspace(0, MAXITER_list[0], MAXITER_list[0]+1 ,endpoint=True),  METHODerror_list[0], label = "Jacobi", linestyle='--',marker='o' )
            plt.plot(np.linspace(0, MAXITER_list[1], MAXITER_list[1]+1 ,endpoint=True),  METHODerror_list[1], label = "Gauss-Seidel", linestyle='--',marker='+')
            plt.plot(np.linspace(0, MAXITER_list[2], MAXITER_list[2]+1 ,endpoint=True),  METHODerror_list[2], label = "SOR", linestyle=':',marker='*')

            plt.legend()
            plt.show()



    
    

    