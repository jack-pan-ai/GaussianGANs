from numpy.random import multivariate_normal
import numpy as np
import pickle
import os
from torch.utils.data import Dataset
import torch

class MultiNormaldataset(Dataset):
    def __init__(self, latent_dim = 64, size = 10000, mode = 'train'):
        if mode == 'train':
            self.simulate_multi_normal(latent_dim, size)
            with open("./MultiNormalDataset/train/multinormal_data.pkl", 'rb') as f:
                self.x = pickle.load(f)
        elif mode == 'test':
            self.simulate_multi_normal(latent_dim, size = size, mode=mode)
            with open('./MultiNormalDataset/test/multinormal_data.pkl', 'rb') as f:
                 self.x = pickle.load(f)
        else:
            raise InterruptedError('Please choose test or train mode')

    def simulate_multi_normal(self, latent_dim, size, mode = 'train'):
        if mode == 'train' :
            if not os.path.exists('./MultiNormalDataset/train/multinormal_data.pkl'):
                if not os.path.exists('./MultiNormalDataset/train'):
                    os.makedirs('MultiNormalDataset/train')
                mean = np.zeros(latent_dim, dtype=float)
                cov = np.diag(np.ones(latent_dim, dtype=float))

                x = multivariate_normal(mean=mean, cov=cov, size=size)
                x = x.reshape(-1, 1, latent_dim)
                with open('./MultiNormalDataset/train/multinormal_data.pkl', 'wb') as f:
                    pickle.dump(x, f)
                print('Simulation for train dataset finished! and the shape is ', x.shape)
            else:
                print("Train Dataset exist!")
        else:
            if not os.path.exists('./MultiNormalDataset/test/multinormal_data.pkl'):
                if not os.path.exists('./MultiNormalDataset/test'):
                    os.makedirs('MultiNormalDataset/test')
                mean = np.zeros(latent_dim, dtype=float)
                cov = np.diag(np.ones(latent_dim, dtype=float))

                x = multivariate_normal(mean=mean, cov=cov, size=size)
                x = x.reshape(-1, 1, latent_dim)
                with open('./MultiNormalDataset/test/multinormal_data.pkl', 'wb') as f:
                    pickle.dump(x, f)
                print('Simulation for test finished!')
            else:
                print("Test Dataset exist!")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item]

if __name__=='__main__':
    dat = MultiNormaldataset(32, 1000, 'test')
    num_sh = torch.randint(0, len(dat), [128,])
    print(dat[num_sh].shape)
    print(dat[:10].shape)

