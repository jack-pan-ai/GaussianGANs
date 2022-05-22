from numpy.random import multivariate_normal
import numpy as np
import pickle
import os
from torch.utils.data import Dataset
import torch

def transform_Gaussian(latent_dim, size, transform, truncate):
    mean = np.random.uniform(1, 10, size=latent_dim)
    # let covariance matrix to be positive-semidefinite
    cov = np.random.uniform(-3, 3, size=latent_dim ** 2).reshape(latent_dim, latent_dim)
    cov = np.dot(cov, cov.T)
    cov = cov + cov.T
    x = multivariate_normal(mean=mean, cov=cov, size=size)

    if transform:
        # non-linear transformation
        x = 2 * np.log(np.abs(x)) + 1
    if truncate:
        # truncation
        c = 7.
        x[x >= c] = c
    # reshape the dataset
    x = x.reshape(-1, 1, latent_dim)

    return x



class MultiNormaldataset(Dataset):
    def __init__(self, latent_dim = 64, size = 10000, mode = 'train', transform = False, truncate = False):
        if mode == 'train':
            if not os.path.exists('./MultiNormalDataset/train'):
                os.makedirs('MultiNormalDataset/train')
            if (not transform and not truncate):
                self.simulate_multi_normal(latent_dim, size, transform = False, truncate = False)
                with open("./MultiNormalDataset/train/multinormal_data" +".pkl", 'rb') as f:
                    self.x = pickle.load(f)
            elif (transform and truncate):
                self.simulate_multi_normal(latent_dim, size, transform=True, truncate=True)
                with open("./MultiNormalDataset/train/multinormal_data" + str('_transform') + str('_truncate') + ".pkl", 'rb') as f:
                    self.x = pickle.load(f)
            elif (transform):
                self.simulate_multi_normal(latent_dim, size, transform=True, truncate=False)
                with open("./MultiNormalDataset/train/multinormal_data" + str('_transform') + ".pkl", 'rb') as f:
                    self.x = pickle.load(f)
            else:
                self.simulate_multi_normal(latent_dim, size, transform=False, truncate=True)
                with open("./MultiNormalDataset/train/multinormal_data" + str('_truncate') + ".pkl", 'rb') as f:
                    self.x = pickle.load(f)
        elif mode == 'test':
            if not os.path.exists('./MultiNormalDataset/test'):
                os.makedirs('MultiNormalDataset/test')
            if (not transform and not truncate):
                self.simulate_multi_normal(latent_dim, size, mode ='test', transform=False, truncate=False)
                with open("./MultiNormalDataset/test/multinormal_data" + ".pkl", 'rb') as f:
                    self.x = pickle.load(f)
            elif (transform and truncate):
                self.simulate_multi_normal(latent_dim, size, mode ='test', transform=True, truncate=True)
                with open("./MultiNormalDataset/test/multinormal_data" + str('_transform') + str('_truncate') + ".pkl", 'rb') as f:
                    self.x = pickle.load(f)
            elif (transform):
                self.simulate_multi_normal(latent_dim, size, mode ='test', transform=True, truncate=False)
                with open("./MultiNormalDataset/test/multinormal_data" + str('_transform') + ".pkl", 'rb') as f:
                    self.x = pickle.load(f)
            else:
                self.simulate_multi_normal(latent_dim, size, mode ='test', transform=False, truncate=True)
                with open("./MultiNormalDataset/test/multinormal_data" + str('_truncate') + ".pkl", 'rb') as f:
                    self.x = pickle.load(f)
        else:
            raise InterruptedError('Please choose test or train mode')

    def simulate_multi_normal(self, latent_dim, size, mode = 'train', transform=False, truncate=False):
        if mode == 'train':
            if (not transform and not truncate):
                if not os.path.exists('./MultiNormalDataset/train/multinormal_data.pkl'):
                    mean = np.zeros(latent_dim, dtype=float)
                    cov = np.diag(np.ones(latent_dim, dtype=float))

                    x = multivariate_normal(mean=mean, cov=cov, size=size)
                    x = x.reshape(-1, 1, latent_dim)
                    with open('./MultiNormalDataset/train/multinormal_data.pkl', 'wb') as f:
                        pickle.dump(x, f)
                    print('(Standard Multinormal Dataset) Simulation for train dataset finished! and the shape is ', x.shape)
                else:
                    print("(Standard Multinormal Dataset) Train Dataset exist!")
            elif (transform and truncate):
                if not os.path.exists('./MultiNormalDataset/train/multinormal_data' + str('_transform') + str('_truncate') +'.pkl'):
                    #mean = np.zeros(latent_dim, dtype=float)
                    x = transform_Gaussian(latent_dim, size, transform, truncate)
                    with open('./MultiNormalDataset/train/multinormal_data' + str('_transform') + str('_truncate') + '.pkl', 'wb') as f:
                        pickle.dump(x, f)
                    print('(Transformed and Truncated Standard Multinormal Dataset) Simulation for train dataset finished! and the shape is ',
                          x.shape)
                else:
                    print("(Transformed and Truncated Standard Multinormal Dataset) Train Dataset exist!")
            elif (transform):
                if not os.path.exists('./MultiNormalDataset/train/multinormal_data' + str('_transform') +'.pkl'):
                    #mean = np.zeros(latent_dim, dtype=float)
                    x = transform_Gaussian(latent_dim, size, transform, truncate)
                    with open('./MultiNormalDataset/train/multinormal_data' + str('_transform') + '.pkl', 'wb') as f:
                        pickle.dump(x, f)
                    print('(Transformed Standard Multinormal Dataset) Simulation for train dataset finished! and the shape is ',
                          x.shape)
                else:
                    print("(Transformed Standard Multinormal Dataset) Train Dataset exist!")
            else:
                if not os.path.exists('./MultiNormalDataset/train/multinormal_data' + str('_truncate') +'.pkl'):
                    #mean = np.zeros(latent_dim, dtype=float)
                    x = transform_Gaussian(latent_dim, size, transform, truncate)
                    with open('./MultiNormalDataset/train/multinormal_data' + str('_truncate') + '.pkl', 'wb') as f:
                        pickle.dump(x, f)
                    print('(Truncated Standard Multinormal Dataset) Simulation for train dataset finished! and the shape is ',
                          x.shape)
                else:
                    print("(Truncated Standard Multinormal Dataset) Train Dataset exist!")
        else:
            if (not transform and not truncate):
                if not os.path.exists('./MultiNormalDataset/test/multinormal_data.pkl'):
                    mean = np.zeros(latent_dim, dtype=float)
                    cov = np.diag(np.ones(latent_dim, dtype=float))

                    x = multivariate_normal(mean=mean, cov=cov, size=size)
                    x = x.reshape(-1, 1, latent_dim)
                    with open('./MultiNormalDataset/test/multinormal_data.pkl', 'wb') as f:
                        pickle.dump(x, f)
                    print('(Standard Multinormal Dataset) Simulation for test dataset finished! and the shape is ',
                          x.shape)
                else:
                    print("(Standard Multinormal Dataset) Test Dataset exist!")
            elif (transform and truncate):
                if not os.path.exists(
                        './MultiNormalDataset/test/multinormal_data' + str('_transform') + str('_truncate') + '.pkl'):
                    x = transform_Gaussian(latent_dim, size, transform, truncate)
                    with open('./MultiNormalDataset/test/multinormal_data' + str('_transform') + str('_truncate') + '.pkl', 'wb') as f:
                        pickle.dump(x, f)
                    print(
                        '(Transformed and Truncated Standard Multinormal Dataset) Simulation for test dataset finished! and the shape is ',
                        x.shape)
                else:
                    print("(Transformed and Truncated Standard Multinormal Dataset) Test Dataset exist!")
            elif (transform):
                if not os.path.exists('./MultiNormalDataset/test/multinormal_data' + str('_transform') + '.pkl'):
                    x = transform_Gaussian(latent_dim, size, transform, truncate)
                    with open('./MultiNormalDataset/test/multinormal_data' + str('_transform') + '.pkl', 'wb') as f:
                        pickle.dump(x, f)
                    print(
                        '(Transformed Standard Multinormal Dataset) Simulation for test dataset finished! and the shape is ',
                        x.shape)
                else:
                    print("(Transformed Standard Multinormal Dataset) Test Dataset exist!")
            else:
                if not os.path.exists('./MultiNormalDataset/test/multinormal_data' + str('_truncate') + '.pkl'):
                    x = transform_Gaussian(latent_dim, size, transform, truncate)
                    with open('./MultiNormalDataset/test/multinormal_data' + str('_truncate') + '.pkl', 'wb') as f:
                        pickle.dump(x, f)
                    print(
                        '(Truncated Standard Multinormal Dataset) Simulation for test dataset finished! and the shape is ',
                        x.shape)
                else:
                    print("(Truncated Standard Multinormal Dataset) Test Dataset exist!")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item]


if __name__=='__main__':
    dat = MultiNormaldataset(32, 1000, 'test')
    num_sh = torch.randint(0, len(dat), [128,])
    print(dat[num_sh].shape)
    print(dat[:10].shape)

