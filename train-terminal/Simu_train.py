#!/usr/bin/env bash

import os


os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../train_GAN.py \
--gpu 0 \
--dist-url 'tcp://localhost:4321' \
--world-size 1 \
--dataset Simulation \
--transform True \
--gen_model my_gen \
--dis_model my_dis \
--heads 5 \
--noise_dim 16 \
--simu_dim 64 \
--simu_channels 4 \
--num_workers 16 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--loss lsgan \
--n_dis 1 \
--n_gen 2 \
--patch_size 15 \
--phi 1 \
--batch_size 256 \
--epochs 1500 \
--eval_num 5000 \
--eval_epochs 5 \
--print_freq 50 \
--checkpoint_best_PATH '/home/panq/python_project/GaussianGANs/train-terminal/save/Ch_4_Dim_64/checkpoint_best_dis' \
--exp_name GANs-Simu-Trans")