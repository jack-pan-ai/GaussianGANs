from numpy.random import multivariate_normal
import numpy as np
import pickle
import os
from torch.utils.data import Dataset
import torch

def _simu_transform_Gaussian(latent_dim, size, transform, truncate, mode, channels=1):

    data_path = './MultiNormalDataset/' + mode + \
                '/data' + ('_transform' if transform else '') +  \
                ('_truncate' if truncate else '') + ('_Dim' + str(latent_dim)) + \
                ('_Chan' + str(channels)) +'.pkl'

    if not os.path.exists(data_path):
        length_whole = latent_dim * channels
        if (not transform and not truncate):
            mean = np.zeros(length_whole, dtype=float)
            cor = np.diag(np.ones(length_whole, dtype=float))
        else:
            if mode == 'train':
                mean = np.random.uniform(1, 3, size=length_whole)
                # let covariance matrix to be positive-semidefinite
                cov = np.random.uniform(-1, 1, size=length_whole ** 2).reshape(length_whole, length_whole)
                cov = np.dot(cov, cov.T)
                cov = cov + cov.T
                var = np.diag(1 / np.sqrt(np.diag(cov)))
                cor = np.matmul(var, cov)
                cor = np.matmul(cor, var)
            else:
                _data_path = data_path.replace('test', 'train')
                with open(_data_path, 'rb') as f:
                    data_GRF_train = pickle.load(f)
                    mean, cor = data_GRF_train['mean'], data_GRF_train['cor']
        x = multivariate_normal(mean=mean, cov=cor, size=size)

        if transform:
            # non-linear transformation
            x = np.exp(x) + 1
        if truncate:
            # truncation (0, +inf)
            c = max(np.abs(x)) / 1.2
            x[x >= c] = c
        # reshape the dataset
        x = x.reshape(-1, channels, latent_dim)

        data_GRF = {'x': x, 'mean': mean, 'cor': cor}

        with open(data_path, 'wb') as f:
            pickle.dump(data_GRF, f)
        print('Simulation for ' + mode + ' dataset finished!' + ' Path: ' + data_path + ' Shape:', x.shape)

        return x, mean, cor
    else:
        with open(data_path, 'rb') as f:
            data_GRF = pickle.load(f)
        x, mean, cor = data_GRF['x'], data_GRF['mean'], data_GRF['cor']
        if x.shape[0] > size:
            no_shuffle = np.random.randint(0, x.shape[0], size)
            x = x[no_shuffle]
        print('Dataset exists: ' + data_path)
        print(mode + ' Shape: ', x.shape)
        return x, mean, cor



class MultiNormaldataset(Dataset):
    def __init__(self, latent_dim, size, mode, channels=None, simu_dim=None, transform=False, truncate=False, appendix=None):
        assert mode == 'train' or mode == 'test', 'Please input the right mode: train or test.'
        if not os.path.exists('./MultiNormalDataset/' + mode):
            os.makedirs('MultiNormalDataset/' + mode)
        if (not transform and not truncate):
            self.x, self.mean, self.cor = _simu_transform_Gaussian(latent_dim, size, transform, truncate, mode, channels)
        else:
            self.x, self.mean, self.cor = _simu_transform_Gaussian(simu_dim, size, transform, truncate, mode, channels)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item]



