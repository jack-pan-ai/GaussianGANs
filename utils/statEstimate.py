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
    # normality
    count = 0
    if x.shape[0] > 4900:
        pvalues_s = [stats.shapiro(x[:4900, i]) for i in range(n_fea)]
    else:
        pvalues_s = [stats.shapiro(x[:, i]) for i in range(n_fea)]
    pvalues_da = [stats.normaltest(x[:, i]) for i in range(n_fea)]
    for pvs, pvd in zip(pvalues_s, pvalues_da):
        if pvs[1] > 0.95 or pvd[1] > 0.95:
            count += 1
    p_dis = count / n_fea
    # correlation matrix
    p_cor = np.corrcoef(x.T)
    cor_dis = np.sqrt(np.sum((p_cor - np.diag(np.ones(n_fea)))**2)) / np.sqrt(n_fea)
    # variance and mean: moment dis
    moment_dis = np.sqrt((x_mean - 0)**2 + (x_std - 1)**2)/ n_fea
    # compute distance
    dis = p_dis + cor_dis + moment_dis

    return dis, p_dis, cor_dis, moment_dis

def heatmap_cor(x, epoch, args = None, save_name=None):
    '''
    used to plot the correlation matrix
    :param x: input batch_size model [batch_size, noise_dimension]
    :return: figure
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
    pear_cor = np.abs(pear_cor)
    fig, ax = plt.subplots()
    sns.heatmap(pear_cor, vmin=0, vmax=1)
    ax.set_title('Heatmap for correlation matrix')
    #plt.show()
    if args is not None:
        plt.savefig(os.path.join(args.path_helper['log_path_img_heatmap'],f'{save_name}_epoch_{epoch+1}.png'), format="png")

def qqplot(x, epoch, args = None, save_name=None):
    '''
    Used to plot the QQ plot for x and chi-square distribution
    :param x: tested data
    :param epoch: index, a monitor for training process
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
    n_fea = x.shape[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.sum(x**2, axis=1)
    res = stats.probplot(x, dist=stats.chi2, sparams=(n_fea,), plot=ax)
    ax.set_title('QQ plot for Chi-square distribution')
    #plt.show()
    if args is not None:
        plt.savefig(os.path.join(args.path_helper['log_path_img_qqplot'],f'{save_name}_epoch_{epoch+1}.png'), format="png")


