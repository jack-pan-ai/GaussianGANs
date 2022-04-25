from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

def qqplot(x):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = stats.probplot(x, dist=stats.chi2, sparams=(64,), plot=ax)
    ax.set_title('QQ plot for Chi-square distribution')
    plt.show()

import sys
print(sys.path)

if __name__=='__main__':
    with open('../inverse_jump.csv', 'r') as f:
        x = np.loadtxt(f, delimiter=',')
        x = np.sum(x**2, axis=0)
    with open('../inv.csv') as f:
        tr = np.loadtxt(f, delimiter=',')
        tr = np.sum(tr**2, axis=0)
    qqplot(x)
    qqplot(tr)
