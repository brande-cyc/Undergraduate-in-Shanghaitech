import numpy as np

def data_loader():
    data = np.array(np.loadtxt('../data_problem1/data.txt'))
    return data

def label_loader():
    label = np.array(np.loadtxt('../data_problem1/label.txt'))
    return label

def sol_save(addr, X):
    np.savetxt(addr, X, fmt="%.8f")