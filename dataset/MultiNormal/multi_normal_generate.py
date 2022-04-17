from numpy.random import multivariate_normal
import numpy as np
import pickle
import os
from torch.utils.data import Dataset

class MultiNormal(Dataset):
    def __init__(self, size = 10000):
        self.simulate_multi_normal(size = size)
        self.x = pickle.load("./multinormal_data.pkl")

    def simulate_multi_normal(self, size):
        if not os.path.exists('./multinormal_data.pkl'):
            mean = np.zeros(64, dtype=float)
            cov = np.diag(np.ones(64, dtype=float))

            x = multivariate_normal(mean=mean, cov=cov, size=size)
            with open('./multinormal_data.pkl', 'wb') as f:
                pickle.dump(x, f)
        else:


    def __len__(self):
        return

    def __getitem__(self, item):
        pass

