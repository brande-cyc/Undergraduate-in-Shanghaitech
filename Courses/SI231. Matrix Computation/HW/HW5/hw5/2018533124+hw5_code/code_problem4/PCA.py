import numpy as np
import copy
from PIL import Image
import os
import matplotlib.pyplot as plt

data1_filepath = "../data_problem4/data1/"
testim_filepath = "../data_problem4/test_images/"
train_filepath = "../data_problem4/training_images/"

Y_label_test = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]
Y_label_train = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4]

data1_pgm = []
testim_pgm = []
train_pgm = []
for item in os.listdir(data1_filepath):
    if '.pgm' in item:
        data1_pgm.append(item)
for item in os.listdir(testim_filepath):
    if '.pgm' in item:
        testim_pgm.append(item)
for item in os.listdir(train_filepath):
    if '.pgm' in item:
        train_pgm.append(item)

def read_img(path):
    im = Image.open(path)    
    return np.array(im)

def convert_img(arr, showImg = False):
    im = Image.fromarray(arr)
    if showImg:
        im.show()
    return im

def save_img(im, path):
    im = im.convert('RGB')
    im.save(path)

def mean_face_func(X_hat):
    meanFace = copy.deepcopy(X_hat[:,0])
    for i in range(1, X_hat.shape[1]):
        meanFace += X_hat[:,i]
    meanFace /= X_hat.shape[1]
    return meanFace

def PCA_algo(X_hat, d):
    X = copy.deepcopy(X_hat)
    meanFace = mean_face_func(X_hat)
    X = X - np.matmul(meanFace.reshape(-1,1), np.ones(X_hat.shape[1]).reshape(1,-1))
    U, SIG, VT = np.linalg.svd(X, full_matrices=False)
    #print(np.matmul(U, U.T))
    # print(meanFace)
    # print("mean of U: {}".format(np.mean(U)))
    # print("mean of X: {}".format(np.mean(X)))
    # print("singular values: {}".format(SIG))
    Y = np.matmul(np.transpose(U[:,0:d]), X)
    return meanFace, U, Y, SIG





if __name__ == "__main__":
    P1flag = False  # see the result of problem 1
    P2flag = False   # see the result of problem 2
    P3flag = True  # see the result of problem 3
    if P1flag:
        allimgs = np.zeros((48*42, len(data1_pgm)))
        col = 0
        for pgm in data1_pgm:
            arr = read_img(data1_filepath + pgm).reshape(-1,)
            allimgs[:,col] = copy.deepcopy(arr/255.0)
            col += 1
        meanFace, U, Y, SIG = PCA_algo(allimgs, 10)
        meanFace = meanFace.reshape(48,42)
        meanIm = convert_img(meanFace*255.0)
        save_img(meanIm,'./figures/problem1/meanface.jpg' )
        eigenFaceList = [1,2,3,10]
        for eigenFaceid in eigenFaceList:
            eigenFace = U[:,eigenFaceid-1].reshape(48,42)
            eigenFace += np.abs(np.min(eigenFace)) * np.ones(eigenFace.shape)
            eigenFace /= np.max(eigenFace)
            print("mean of eigen face {}: {}".format(eigenFaceid, np.mean(eigenFace)))
            eigenIm = convert_img(eigenFace*255.0)
            save_img(eigenIm, './figures/problem1/eigenface_{}.jpg'.format(eigenFaceid))

    elif P2flag:
        allimgs = np.zeros((48*42, len(train_pgm)))
        col = 0
        for pgm in train_pgm:
            arr = read_img(train_filepath + pgm).reshape(-1,)
            allimgs[:,col] = copy.deepcopy(arr/255.0)
            col += 1
        meanFace_train, U_train, Y, SIG = PCA_algo(allimgs, 10)
        
        meanFace_train = meanFace_train.reshape(48,42)
        meanIm = convert_img(meanFace_train*255.0)
        save_img(meanIm,'./figures/problem2/meanface.jpg' )
        eigenFaceList = [1,2,3,10]
        for eigenFaceid in eigenFaceList:
            eigenFace = U_train[:,eigenFaceid-1].reshape(48,42)
            eigenFace += np.abs(np.min(eigenFace)) * np.ones(eigenFace.shape)
            eigenFace /= np.max(eigenFace)
            print("mean of eigen face {}: {}".format(eigenFaceid, np.mean(eigenFace)))
            eigenIm = convert_img(eigenFace*255.0)
            save_img(eigenIm, './figures/problem2/eigenface_{}.jpg'.format(eigenFaceid))
        
        # problem b
        PLOT_FLAG = False
        if PLOT_FLAG:
            plt.title("Sorted singular value for Training Set")
            plt.xlabel('singular values')
            plt.ylabel('value')

            plt.plot(range(1, len(SIG) + 1), SIG)

            plt.show()

        # problem c
        dList = [10,2]
        for d in dList:
            allimgs = np.zeros((48*42, len(train_pgm)))
            col = 0
            for pgm in train_pgm:
                arr = read_img(train_filepath + pgm).reshape(-1,)
                allimgs[:,col] = copy.deepcopy(arr/255.0)
                col += 1
            meanFace_train, U_train, Y, SIG = PCA_algo(allimgs, d)


            testimgs = np.zeros((48*42, len(testim_pgm)))
            col = 0
            for pgm in testim_pgm:
                arr = read_img(testim_filepath + pgm).reshape(-1,)
                testimgs[:,col] = copy.deepcopy(arr/255.0)
                col += 1
            meanFace_test, U_test, Y, SIG = PCA_algo(testimgs, d)
            
            Y_test = np.matmul(np.transpose(U_train[:,0:d]), testimgs - np.matmul(meanFace_train.reshape(-1,1), np.ones(testimgs.shape[1]).reshape(1,-1)))
            ProjXtest = np.matmul(meanFace_train.reshape(-1,1), np.ones(testimgs.shape[1]).reshape(1,-1)) + np.matmul(U_train[:,0:d], Y_test)
            
            plotid = 15
            plotFace = ProjXtest[:,plotid].reshape(48,42)
            plotFace += np.abs(np.min(plotFace)) * np.ones(plotFace.shape)
            plotFace /= np.max(plotFace)
            plotIm = convert_img(plotFace*255.0)
            save_img(plotIm, './figures/problem2/projectface_{}_d-{}.jpg'.format(plotid, d))
    
    elif P3flag:
        dList = range(2,9)
        errorRateList = []
        for d in dList:
            trainimgs = np.zeros((48*42, len(train_pgm)))
            col = 0
            for pgm in train_pgm:
                arr = read_img(train_filepath + pgm).reshape(-1,)
                trainimgs[:,col] = copy.deepcopy(arr/255.0)
                col += 1
            meanFace_train, U_train, Y, SIG = PCA_algo(trainimgs, d)
            ProjXtrain = np.matmul(meanFace_train.reshape(-1,1), np.ones(trainimgs.shape[1]).reshape(1,-1)) + np.matmul(U_train[:,0:d], Y)

            testimgs = np.zeros((48*42, len(testim_pgm)))
            col = 0
            for pgm in testim_pgm:
                arr = read_img(testim_filepath + pgm).reshape(-1,)
                testimgs[:,col] = copy.deepcopy(arr/255.0)
                col += 1
            meanFace_test, U_test, Y, SIG = PCA_algo(testimgs, d)
            Y_test = np.matmul(np.transpose(U_train[:,0:d]), testimgs - np.matmul(meanFace_train.reshape(-1,1), np.ones(testimgs.shape[1]).reshape(1,-1)))
            ProjXtest = np.matmul(meanFace_train.reshape(-1,1), np.ones(testimgs.shape[1]).reshape(1,-1)) + np.matmul(U_train[:,0:d], Y_test)

            labelList = []
            for i in range(len(testim_pgm)):
                imageItest = ProjXtest[:,i]
                dist = float('inf')
                label = 1
                for j in range(len(train_pgm)):
                    imageJtrain = ProjXtrain[:,j]
                    currDist = np.linalg.norm(imageItest-imageJtrain)
                    if currDist < dist:
                        dist = currDist
                        label = Y_label_train[j]
                labelList.append(label)
            
            error_count = 0.0
            for i in range(len(labelList)):
                if labelList[i] != Y_label_test[i]:
                    error_count += 1.0
            errorRateList.append(error_count/len(labelList))
        print(errorRateList)
            

            

