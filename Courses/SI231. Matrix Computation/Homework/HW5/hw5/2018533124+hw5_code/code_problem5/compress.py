import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import scipy.io as scio

matpath = '../data_problem4/data1/data1.mat'
datamat = scio.loadmat(matpath)
data1 = copy.deepcopy(np.array(datamat['data1']))

def compressionRate(d_star, D, N):
    return float(d_star * (D + N + 1)) / float(D * N)

def f_func(d, SquaredEigenV):
    if d == len(SquaredEigenV) - 1:
        print("Exceed, d")
        return 0
    return np.sum(SquaredEigenV[d+1:]) / np.sum(SquaredEigenV)


if __name__ == "__main__":
    Flagp1 = False
    Flagp2 = True

    if Flagp1:
        A = copy.deepcopy(data1)
        K = min(A.shape[0], A.shape[1])
        eigenValsquare = np.linalg.eigvals(np.matmul(A.T, A))
        eigenValsquare = sorted(eigenValsquare, reverse=True)
        currSum = 0
        threshold = 150
        for d in range(len(eigenValsquare)-1, -1, -1):
            currSum += eigenValsquare[d]
            if currSum > 150:
                print(d+1)
                break
        plt.title("Squared singular values of A")
        plt.xlabel('singular values')
        plt.ylabel('value')

        plt.plot(range(1, len(eigenValsquare) + 1), threshold * np.ones(len(eigenValsquare)), color = 'red')
        plt.plot(range(1, len(eigenValsquare) + 1), eigenValsquare)

        plt.show()
    if Flagp2:
        A = copy.deepcopy(data1)
        K = min(A.shape[0], A.shape[1])
        eigenValsquare = np.linalg.eigvals(np.matmul(A.T, A))
        eigenValsquare = sorted(eigenValsquare, reverse=True)
        totalSum = np.sum(eigenValsquare)
        thresholdList = [0.1, 0.05, 0.02, 0.005]
        functionValueList = []
        for d in range(len(eigenValsquare)):
            currVal = f_func(d, eigenValsquare)
            functionValueList.append(currVal)
        print("************************************************")
        plt.title("f(d)")
        plt.xlabel('d')
        plt.ylabel('value')

        plt.step(range(1, len(eigenValsquare) + 1), functionValueList, where='post')

        plt.show()
        for threshold in thresholdList:
            for d in range(len(eigenValsquare)):
                currVal = f_func(d, eigenValsquare)
                if currVal <= threshold:
                    d_star = d + 1
                    print("threshold = {}".format(threshold))
                    print("compression rate = {}".format(compressionRate(d_star, A.shape[0], A.shape[1])))
                    print("=============================================================================")
                    break

        
