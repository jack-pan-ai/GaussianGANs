import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

from dataset.UniMiB.dataLoader import *
from utils.utils import load_checkpoint
from models.inverseGANModels import inverseGenerator

import torch
import numpy as np

# time series dataset [size, 3, 150]
img_jump = unimib_load_dataset(incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=False, is_normalize=True,
                                    one_hot_encode=False, data_mode='Train', single_class=True,
                                    class_name='Jumping', augment_times=None)
img_run = unimib_load_dataset(incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=False, is_normalize=True,
                                    one_hot_encode=False, data_mode='Train', single_class=True,
                                    class_name='Running', augment_times=None)
invGen_net = inverseGenerator(seq_len=150, channels=3,
                           num_heads=5, latent_dim=64,
                           depth=4, patch_size=15).cuda(0)
checkpoint_PATH = 'train-terminal/logs/JumpingInverse_2022_04_19_19_57_42/Model/checkpoint'
load_checkpoint(invGen_net, checkpoint_PATH)

with torch.no_grad():
    img_jump_set = torch.from_numpy(img_jump[:100]).type(torch.cuda.FloatTensor).cuda(0)
    out = invGen_net(img_jump_set)
    print('The shape of data transformed by the inverse Generator: ', out.shape)

# [batch_size, 1, 64] - > [batch_size, 64]
out = out.squeeze(1).detach().cpu().numpy()
np.savetxt('inverse_jump.csv', out, delimiter=',')




