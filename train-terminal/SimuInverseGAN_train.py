#!/usr/bin/env bash

import os
import argparse

os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../train_inverseGAN.py \
--gpu 1 \
--dist-url 'tcp://localhost:4321' \
--world-size 1 \
--dataset Simulation \
--transform True \
--simu_dim 150 \
--simu_channels 3 \
--gen_model my_gen \
--dis_model my_dis \
--heads 5 \
--noise_dim 16 \
--num_workers 16 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--loss lsgan \
--n_dis 4 \
--patch_size 15 \
--phi 1 \
--batch_size 256 \
--epochs 1500 \
--print_freq 50 \
--exp_name GauGANs-Simu-Trans")