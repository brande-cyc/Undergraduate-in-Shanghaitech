import matplotlib.pyplot as plt 
import numpy as np

inv_time = np.load('inv_time.npy')
cramer_time = np.load('cramer_time.npy')
gauss_time = np.load('gauss_time.npy')
lu_time = np.load('lu_time.npy')

ns = np.linspace(100,1000,19)
plt.figure(dpi=400)
plt.plot(ns, inv_time,label = 'Inverse', marker = '>', markevery=1, lw=1)
plt.plot(ns, cramer_time, label = 'Cramer', marker = 's', markevery=1, lw=1)
plt.plot(ns, gauss_time, label = 'G.E', marker = '.', markevery=1, lw=1)
plt.plot(ns, lu_time,label = 'LU', marker = '*', markevery=1, lw=1)
plt.xlabel('Size')
plt.ylabel('Time')
plt.grid()
plt.legend()
plt.savefig('Plot.png')