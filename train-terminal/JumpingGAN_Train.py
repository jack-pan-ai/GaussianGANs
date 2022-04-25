#!/usr/bin/env bash

import os
import argparse


os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ../train_GAN.py \
--gpu 0 \
--dist-url 'tcp://localhost:4321' \
--world-size 1 \
--dataset UniMiB \
--gen_model my_gen \
--dis_model my_dis \
--heads 5 \
--noise_dim 64 \
--num_workers 16 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--loss lsgan \
--n_critic 2 \
--patch_size 15 \
--phi 1 \
--epochs 10000 \
--batch_size 16 \
--print_freq 50 \
--class_name Jumping \
--exp_name Jumping")