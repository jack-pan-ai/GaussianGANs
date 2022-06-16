# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

# @Date    : 2022-06-16
# @Author  : Qilong Pan (qilong.pan@kaust.edu.sa)
# @Link    : None
# @Version : 0.1

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # misc configuration
    parser.add_argument('--seed', default=12345, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='number of cpu threads to use during batch generation')

    # training configuration
    parser.add_argument( '--epochs', type=int, default=777,
                         help='number of epochs of training')
    parser.add_argument('--max_iter', type=int, default=50000,
                        help='set the max iteration number')
    parser.add_argument('-bs', '--batch_size', type=int, default=64,
                        help='size of the batches to load dataset')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='adam: Generator learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='adam: Generator weight decay')
    parser.add_argument('--d_lr', type=float, default=0.0002,
                        help='adam: Discriminator learning rate')
    parser.add_argument('--lr_decay', action='store_true',
                        help='learning rate decay or not')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='adam 1ts order: coefficients used for computing running averages of gradient')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='adam 2nd order: coefficients used for computing running averages of gradient')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='dimensionality of the latent space')
    parser.add_argument('--patch_size', type=int, default=15,
                        help='number of patch size should be integer factor for seq_length')
    parser.add_argument('--n_dis', type=int, default=1,
                        help='number of training steps for discriminator per iter')
    parser.add_argument('--n_gen', type=int, default=1,
                        help='number of training steps for generator per iter')


    # dataset and path configuration
    parser.add_argument('--data_path', type=str, default='./data',
                        help='The path of data set')
    parser.add_argument('--checkpoint_best_PATH', type=str, default=None,
                        help='With well-trained inverse GAN, provide the path of best model')
    parser.add_argument('--load_path', type=str,
                        help='The reload model path')
    parser.add_argument('--class_name', type=str,
                        help='The class name to load in UniMiB dataset')
    parser.add_argument('--augment_times', type=int, default=None,
                        help='The times of augment signals compare to original data')
    parser.add_argument('--exp_name', type=str,
                        help='The name of experiment')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset type')

    # evaluation metrix configuration
    parser.add_argument('--swell_ratio', type = float, default=1.3,
                        help='visulizationMetrics: the ratio that enlarge the size of figure, eg, x_min * swell_ratio')
    parser.add_argument('--eval_epochs', type=int, default=10,
                        help='The evaluation epoch for generated data and save the generated samples')
    parser.add_argument('--eval_num', type=int, default=400,
                       help='The evaluation number for generated data and save the generated samples')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='interval between each verbose')

    # model configuration
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='optimizer')
    parser.add_argument('--loss', type=str, default="standard",
                        help='loss function')
    parser.add_argument('--d_depth', type=int, default=4,
                        help='Discriminator Depth')
    parser.add_argument('--g_depth', type=int, default=4,
                        help='Generator Depth')
    parser.add_argument('--accumulated_times', type=int, default=1,
                        help='gradient accumulation')
    parser.add_argument('--g_accumulated_times', type=int, default=1,
                        help='gradient accumulation')
    parser.add_argument('--heads', type=int, default=5,
                        help='number of heads')
    parser.add_argument('--init_type', type=str, default='xavier_uniform',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='The init type')

    # Simulation setting for Gaussian Random Field and Vector AR(1) model
    parser.add_argument('--transform', type=bool, default=False,
                        help='Transformation on Generated Multivariate Gaussian Distribution')
    parser.add_argument('--truncate', type=bool, default=False,
                        help='Truncate on Generated Multivariate Gaussian Distribution')
    parser.add_argument('--simu_dim', type=int, default=150,
                        help='The dimension of simulated Gaussian Random Fields')
    parser.add_argument('--simu_channels', type=int, default=3,
                        help='The channels of simulated Gaussian Random Fields')

    opt = parser.parse_args()
    return opt