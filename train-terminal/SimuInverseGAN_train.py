#!/usr/bin/env bash

import os
import argparse

os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../train_inverseGAN.py \
--gpu 0 \
--dist-url 'tcp://localhost:4321' \
--world-size 1 \
--dataset Simulation \
--simu_dim 150 \
--simu_channels 3 \
--noise_dim 64 \
--gen_model my_gen \
--dis_model my_dis \
--heads 5 \
--num_workers 8 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--loss lsgan \
--n_dis 1 \
--n_gen 2 \
--d_depth 4 \
--d_depth 4 \
--patch_size 15 \
--phi 1 \
--batch_size 256 \
--eval_epochs 5 \
--epochs 1500 \
--print_freq 50 \
--exp_name GauGANs-VAR")