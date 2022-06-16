#!/usr/bin/env bash

import os

os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../train_GAN.py \
--gpu 0 \
--dataset Simulation \
--transform True \
--noise_dim 16 \
--simu_dim 64 \
--simu_channels 4 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--loss lsgan \
--n_dis 1 \
--n_gen 1 \
--patch_size 15 \
--batch_size 256 \
--epochs 1500 \
--eval_epochs 5 \
--print_freq 50 \
--checkpoint_best_PATH 'save/Ch_4_Dim_16/checkpoint_best_cor' \
--exp_name GANs-Simu-Trans")