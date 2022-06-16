#!/usr/bin/env bash

import os
import argparse

os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../train_inverseGAN.py \
--gpu 0 \
--dataset UniMiB \
--noise_dim 16 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--loss lsgan \
--n_dis 4 \
--patch_size 15 \
--epochs 3000 \
--batch_size 128 \
--print_freq 50 \
--class_name Jumping \
--exp_name JumpingInverse-dis-v3")