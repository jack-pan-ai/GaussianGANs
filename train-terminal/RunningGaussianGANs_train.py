#!/usr/bin/env bash

import os
import argparse

os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../train_GaussianeGAN.py \
--gpu 0 \
--dataset UniMiB \
--latent_dim 16 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--loss lsgan \
--n_dis 4 \
--n_gen 1 \
--patch_size 15 \
--batch_size 128 \
--epochs 3000 \
--print_freq 50 \
--class_name Running \
--exp_name Running-Gau")