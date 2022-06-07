import numpy as np
from numpy.random import multivariate_normal
from einops import rearrange
from statsmodels.tsa.api import VAR
import pickle
import os
from torch.utils.data import Dataset

MEAN = [0,0,0]


def _var_simu(Phi, Omega, time_step, channels, sample_size):
    x0 = np.random.rand(sample_size, channels, 1)
    x = x0
    for t in range(time_step - 1):
        # [batch_size, channels, timestep] -> [batch_size, timestep, channels]
        _xpre = rearrange(x[:, :, [t]], 'b c t -> b t c')
        # phi*x_{t-1}
        _xt = np.matmul(_xpre, Phi.T)
        # [batch_size, timestep, channels] -> [batch_size, channels, timestep]
        _xt = rearrange(_xt, 'b t c -> b c t')
        # x_t = x_{t-1} + epsilon
        _xt = _xt + multivariate_normal(mean=MEAN, cov=Omega, size=1).reshape(1, -1, 1)
        # concate
        x = np.concatenate((_xt, x), axis=2)

    return x

def _var_simu_dataset(latent_dim, size, mode, channels=1):

    data_path = './VARdataset/' + mode + \
                '/data' + ('_Dim' + str(latent_dim)) + \
                ('_Chan' + str(channels)) +'.pkl'

    if not os.path.exists(data_path):
        if mode == 'train':
            # mean -> phi_1
            # PHI = np.array(np.diag([0.9, 0.9, 0.9]))
            mean = np.array([[0.5, 0.2, -0.2],
                              [0.4, 0.3, 0.4],
                              [0.3, 0.5, 0.1]])
            # cor -> OMEGA
            cor = [[5, 1, -2],
                     [1, 3, 1],
                     [-2, 1, 4]]
            # OMEGA = np.diag([1, 1, 1])
        else:
            _data_path = data_path.replace('test', 'train')
            with open(_data_path, 'rb') as f:
                data_GRF_train = pickle.load(f)
                mean, cor = data_GRF_train['mean'], data_GRF_train['cor']
        x = _var_simu(Phi=mean, Omega=cor, time_step=latent_dim, channels=channels, sample_size=size)

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



class VARdataset(Dataset):
    def __init__(self, latent_dim, size, mode, channels=None):
        assert mode == 'train' or mode == 'test', 'Please input the right mode: train or test.'
        if not os.path.exists('./VARdataset/' + mode):
            os.makedirs('VARdataset/' + mode)
        self.x, self.mean, self.cor = _var_simu_dataset(latent_dim, size, mode, channels)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item]





