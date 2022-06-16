#!/usr/bin/env bash

import os
import argparse


os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../train_GAN.py \
--gpu 0 \
--dataset UniMiB \
--latent_dim 16 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--loss lsgan \
--n_dis 1 \
--n_gen 2 \
--patch_size 15 \
--epochs 3000 \
--batch_size 16 \
--print_freq 50 \
--checkpoint_best_PATH '/home/panq/python_project/GaussianGANs/train-terminal/save/JumpingGaussianGANs' \
--class_name Jumping \
--exp_name Jumping-dis-v3-ground")