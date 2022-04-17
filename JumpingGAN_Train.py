#!/usr/bin/env bash

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    opt = parser.parse_args()

    return opt
args = parse_args()

os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_GAN.py \
--gpu 0 \
--dist-url 'tcp://localhost:4321' \
--dist-backend 'nccl' \
--world-size 1 \
--rank {args.rank} \
--dataset UniMiB \
--bottom_width 8 \
--gen_model my_gen \
--dis_model my_dis \
--heads 5 \
--noise_dim 65 \
--num_workers 16 \
--g_lr 0.0001 \
--d_lr 0.0003 \
--loss lsgan \
--n_critic 2 \
--phi 1 \
--epochs 10000 \
--batch_size 16 \
--print_freq 50 \
--diff_aug translation,cutout,color \
--class_name Jumping \
--exp_name Jumping")