from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import os

def diff_cor(x):
    '''
    Used to choose the best generative and inverse generative model using the matrix difference distance
    :param x: input batch_size model [batch_size, noise_dimension]
    :return: distance for determinant and Euclidean
    '''
    if type(x) is torch.Tensor:
        if x.get_device() != -1:
            x = x.to('cpu')
        x = x.numpy()
    elif type(x) is np.ndarray:
        pass
    else:
        print('The generated is not either torch.tensor, nor np.ndarray. Please check your data type for the output of generator.')
        raise TypeError

    n_fea = x.shape[1]
    x_mean = np.sum(np.abs(np.mean(x, axis=0)))
    x_std = np.sum(np.std(x, axis=0))
    pvalue_shapiro = np.mean([stats.shapiro(x[:, i]) for i in range(n_fea)])
    p_cor = np.corrcoef(x.T)
    diff_cor = (p_cor - np.diag(np.ones(n_fea)))
    # compute the determinant value and  Euclidean distance
    eucl_dis = np.mean(diff_cor**2) + np.mean(x_mean) + np.mean(np.abs(x_std - 1.)) - pvalue_shapiro

    return eucl_dis

def heatmap_cor(x, epoch, args = None):
    '''
    used to plot the correlation matrix
    :param x: input batch_size model [batch_size, noise_dimension]
    :return: figrue
    '''
    if type(x) is torch.Tensor:
        if x.get_device:
            x = x.cpu()
        x = x.numpy()
    elif type(x) is np.ndarray:
        pass
    else:
        print('The generated is not either torch.tensor, nor np.ndarray. Please check your data type for the output of generator.')
        raise TypeError

    pear_cor = np.corrcoef(x.T)
    fig, ax = plt.subplots()
    sns.heatmap(pear_cor, vmin=0, vmax=1)
    ax.set_title('Heatmap for correlation matrix')
    #plt.show()
    if args is not None:
        plt.savefig(os.path.join(args.path_helper['log_path_img_heatmap'],f'{args.exp_name}_epoch_{epoch+1}.png'), format="png")


def qqplot(x, epoch, args = None):
    if type(x) is torch.Tensor:
        if x.get_device:
            x = x.cpu()
        x = x.numpy()
    elif type(x) is np.ndarray:
        pass
    else:
        print('The generated is not either torch.tensor, nor np.ndarray. Please check your data type for the output of generator.')
        raise TypeError
    n_fea = x.shape[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.sum(x**2, axis=0)
    res = stats.probplot(x, dist=stats.chi2, sparams=(n_fea,), plot=ax)
    ax.set_title('QQ plot for Chi-square distribution')
    #plt.show()
    if args is not None:
        plt.savefig(os.path.join(args.path_helper['log_path_img_qqplot'],f'{args.exp_name}_epoch_{epoch+1}.png'), format="png")


if __name__=='__main__':
    with open('../inverse_jump.csv', 'r') as f:
        x = np.loadtxt(f, delimiter=',')
        #x = np.sum(x**2, axis=1)
    with open('../inv.csv') as f:
        tr = np.loadtxt(f, delimiter=',')
        #tr = np.sum(tr**2, axis=1)
    print(diff_cor(tr))
    print(diff_cor(x))
    heatmap_cor(x)

