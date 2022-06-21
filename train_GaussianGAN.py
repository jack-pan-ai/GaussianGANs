from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import cfg
from dataset.UniMiB.dataLoader import *
from models.GaussianGANModels import *
from utils.functions import inverse_train, LinearLrDecay
from utils.utils import set_log_dir, save_checkpoint, create_logger
from utils.visualizationMetrics import visualization
from dataset.MultiNormal.multi_normal_generate import MultiNormaldataset
from utils.statEstimate import *

import torch
import torch.utils.data.distributed
import os
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import random
import matplotlib.pyplot as plt
from torchinfo import summary

# synthesis data
import wandb


def main():
    args = cfg.parse_args()

    if args.seed is not None:
        # reproduce the results
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False  # the accelerator for CNN computation
        torch.backends.cudnn.deterministic = True
        main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    #-------------------------------------------------------------------#
    #------------------------   dataset loading ------------------------#
    # -------------------------------------------------------------------#

    # [batch_size, latent dim]
    train_set = MultiNormaldataset(latent_dim=args.latent_dim, size=10000, mode = 'train', channels=1)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    # test_set = MultiNormaldataset(latent_dim=args.latent_dim, size = 1000, mode='test', simu_dim=args.simu_dim)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    # [batch_size, channles, seq-len]
    print('------------------------------------------------------------')
    print('--------------------GaussianGANs Dataset (Verification in visualization)--------------------------')
    if args.dataset =='UniMiB':
        img_set = unimib_load_dataset(incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=False, is_normalize=True,
                                        one_hot_encode=False, data_mode='Train', single_class=True,
                                        class_name=args.class_name, augment_times=args.augment_times)
        ver_set = unimib_load_dataset(incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=False,
                                      is_normalize=True,
                                      one_hot_encode=False, data_mode='Test', single_class=True,
                                      class_name=args.class_name, augment_times=args.augment_times)
    elif args.dataset =='Simulation':
        img_set = MultiNormaldataset(
            latent_dim=args.latent_dim, size=10000,  # large enough to get a approximate sampling space
            mode='train', channels=args.simu_channels,
            simu_dim=args.simu_dim,
            transform=args.transform, truncate=args.truncate)
        ver_set = MultiNormaldataset(latent_dim=args.latent_dim, size=args.eval_num, mode='test',
                                      channels=args.simu_channels, transform=args.transform,
                                      truncate=args.truncate, simu_dim=args.simu_dim)
    else:
        print('Please input the correct dataset name: UniMiB or Simulation.')
        raise TypeError
    print('------------------------------------------------------------')
    print('How many iterations in a training epoch: ', len(train_loader))
    print('------------------------------------------------------------')
    assert len(train_set) >= args.eval_num, 'The number of evaluation should be less than test_set'

    #-------------------------------------------------------------------#
    #------------------------   Model & optimizer init ------------------#
    # -------------------------------------------------------------------#

    gen_channels, gen_seq_len = img_set[1].shape
    gen_net = inverseGenerator(seq_len=gen_seq_len, channels=gen_channels,
                        num_heads=args.heads, latent_dim=args.latent_dim,
                        depth=args.g_depth, patch_size=args.patch_size)
    dis_channels, dis_seq_len = train_set[1].shape
    dis_net = inverseDiscriminator(seq_len=dis_seq_len, channels=dis_channels,
                            num_heads=args.heads, depth=args.d_depth)
                            #patch_size=args.patch_size)
    torch.cuda.set_device(args.gpu)
    gen_net.cuda(args.gpu)
    dis_net.cuda(args.gpu)
    # model size for generator
    summary(gen_net, (args.batch_size, gen_channels, gen_seq_len))

    # set optimizer
    if args.optimizer == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                         args.g_lr, (args.beta1, args.beta2), weight_decay=args.wd)
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.d_lr, (args.beta1, args.beta2), weight_decay=args.wd)
    else:
        '''
        TO DO: add other optimizer
        '''
        pass
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_gen)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_dis)

    # global setting
    start_epoch = 0
    dis_best, p_dis_best, cor_dis_best, moment_dis_best = 99, 99, 99, 99

    #-------------------------------------------------------------------#
    #------------------------ Writer Configuration  --------------------#
    # -------------------------------------------------------------------#

    # set writer
    assert args.exp_name
    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    writer = SummaryWriter(args.path_helper['log_path'])
    logger.info(args)
    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch,
    }
    print('------------------------------------------------------------')
    print(f"Log file path: {args.path_helper['prefix']}")
    print('------------------------------------------------------------')

    # wandb ai monitoring
    if args.dataset == 'Simulation':
        project_name = 'n_gen: ' + str(args.n_gen) + ', n_dis: ' + str(args.n_dis) + ', ' + \
                       str(args.simu_channels)+ '*'+ str(args.simu_dim) + \
                       (str('trans') if args.transform else '') + (str('trun') if args.truncate else '')
    else:
        project_name = 'n_gen: ' + str(args.n_gen) + ', n_dis: ' + str(args.n_dis) + ', ' + str(args.latent_dim)
    wandb.init(project=args.dataset + args.exp_name, entity="qilong77", name=project_name)
    wandb.config = {
        "epochs": int(args.epochs) - int(start_epoch),
        "batch_size": args.batch_size
    }

    # train loop
    for epoch in range(int(start_epoch), int(args.epochs)):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        inverse_train(args, gen_net, dis_net, gen_optimizer, dis_optimizer,
              train_loader, epoch, writer_dict, img_set, lr_schedulers)

        # save the generated time series after using PCA and t-SNE
        gen_net.eval()
        dis_net.eval()
        if (epoch) % args.eval_epochs == 0:
            with torch.no_grad():
                se_no = torch.randint(0, len(img_set), [args.eval_num])
                img_s = torch.from_numpy(img_set[se_no]).type(torch.cuda.FloatTensor).cuda(args.gpu)
                gen_noise = gen_net(img_s).detach().to('cpu')
            if args.dataset == 'Simulation':
                ver_s = torch.from_numpy(ver_set[:]).type(torch.cuda.FloatTensor).cuda(args.gpu)
                gen_noise_ver = gen_net(ver_s).detach().to('cpu')

            # visuliztion: pca or t-sne plot or heatmap or qqplot
            visualization(ori_data=train_set[:args.eval_num], generated_data=gen_noise[:args.eval_num], analysis='pca',
                          save_name=args.exp_name, epoch=epoch, args=args)
            qqplot(gen_noise.squeeze(1), epoch=epoch, args=args, save_name=args.exp_name)
            heatmap_cor(gen_noise.squeeze(1), epoch=epoch, args=args, save_name=args.exp_name)

            if args.dataset == 'Simulation':
                # verification
                name_ground_truth = args.exp_name + str('ground')
                # visuliztion: pca or t-sne plot or heatmap or qqplot
                visualization(ori_data=train_set[:args.eval_num],
                              generated_data=gen_noise_ver[:args.eval_num],
                              analysis='pca',
                              save_name=name_ground_truth, epoch=epoch, args=args)
                qqplot(gen_noise_ver[:args.eval_num].squeeze(1),
                       epoch=epoch, args=args, save_name=name_ground_truth)
                heatmap_cor(gen_noise_ver[:args.eval_num].squeeze(1),
                            epoch=epoch, args=args, save_name=name_ground_truth)
                dis_ground, p_dis_ground, cor_dis_ground, moment_dis_ground = diff_cor(gen_noise_ver.squeeze(1))

            # correlation matrix distance
            if epoch < 100:  # Used as burnin
                is_best_dis, is_best_p, is_best_cor, is_best_moment = False, False, False, False
                dis, p_dis, cor_dis, moment_dis = diff_cor(gen_noise.squeeze(1))
            else:
                dis, p_dis, cor_dis, moment_dis = diff_cor(gen_noise.squeeze(1))
                if dis < dis_best:
                    dis_best = dis
                    is_best_dis = True
                if p_dis <= p_dis_best:
                    p_dis_best = p_dis
                    is_best_p = True
                if cor_dis < cor_dis_best:
                    cor_dis_best = cor_dis
                    is_best_cor = True
                if moment_dis < moment_dis_best:
                    moment_dis_best = moment_dis
                    is_best_moment = True

            visu_pca = plt.imread(
                os.path.join(args.path_helper['log_path_img_pca'], f'{args.exp_name}_epoch_{epoch + 1}.png'))
            visu_qqplot = plt.imread(
                os.path.join(args.path_helper['log_path_img_qqplot'], f'{args.exp_name}_epoch_{epoch + 1}.png'))
            visu_heatmap = plt.imread(
                os.path.join(args.path_helper['log_path_img_heatmap'], f'{args.exp_name}_epoch_{epoch + 1}.png'))
            if args.dataset == 'Simulation':
                visu_pca_ground = plt.imread(
                    os.path.join(args.path_helper['log_path_img_pca'], f'{name_ground_truth}_epoch_{epoch + 1}.png'))
                visu_qqplot_ground = plt.imread(
                    os.path.join(args.path_helper['log_path_img_qqplot'], f'{name_ground_truth}_epoch_{epoch + 1}.png'))
                visu_heatmap_ground = plt.imread(
                    os.path.join(args.path_helper['log_path_img_heatmap'],
                                 f'{name_ground_truth}_epoch_{epoch + 1}.png'))

            # wandb monitor
            img_visu_pca = wandb.Image(visu_pca, caption="Epoch: " + str(epoch))
            img_visu_qqplot = wandb.Image(visu_qqplot, caption="Epoch: " + str(epoch))
            img_visu_heatmap = wandb.Image(visu_heatmap, caption="Epoch: " + str(epoch))
            if args.dataset == 'Simulation':
                img_visu_pca_ground = wandb.Image(visu_pca_ground, caption="Epoch: " + str(epoch))
                img_visu_qqplot_ground = wandb.Image(visu_qqplot_ground, caption="Epoch: " + str(epoch))
                img_visu_heatmap_ground = wandb.Image(visu_heatmap_ground, caption="Epoch: " + str(epoch))
            wandb.log({'PCA': img_visu_pca,
                       'QQplot': img_visu_qqplot,
                       'Heatmap': img_visu_heatmap})
            if args.dataset == 'Simulation':
                wandb.log({'Verification PCA': img_visu_pca_ground,
                           'Verification QQplot': img_visu_qqplot_ground,
                           'Verification Heatmap': img_visu_heatmap_ground})
            wandb.log({'Distance': dis,
                       'p-value': p_dis,
                       'cor_distance': cor_dis,
                       'moment_dis': moment_dis})
            if args.dataset == 'Simulation':
                wandb.log({'Verification_Distance': dis_ground,
                           'Verification_p-value': p_dis_ground,
                           'Verification_cor_distance': cor_dis_ground,
                           'Verification_moment_dis': moment_dis_ground,
                           })
            # visu_pca = rearrange(visu_pca, 'h w c -> c h w')
            # writer.add_image('Comparison for original and generative data based on PCA', visu_pca, epoch + 1)
            # writer.add_scalar('Correlation matrix distance using determinant', dis)
            # writer.add_scalar('Correlation matrix distance using Euclidean distance', eucl_dis)

        # save the best model
        if epoch >= 100:
            if is_best_dis:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'gen_model': args.gen_model,
                    'dis_model': args.dis_model,
                    'gen_state_dict': gen_net.state_dict(),
                    'dis_state_dict': dis_net.state_dict(),
                    'gen_optimizer': gen_optimizer.state_dict(),
                    'dis_optimizer': dis_optimizer.state_dict(),
                    'path_helper': args.path_helper,
                }, args.path_helper['ckpt_path'], filename="checkpoint_best_dis")
                is_best_dis = False
            elif is_best_moment:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'gen_model': args.gen_model,
                    'dis_model': args.dis_model,
                    'gen_state_dict': gen_net.state_dict(),
                    'dis_state_dict': dis_net.state_dict(),
                    'gen_optimizer': gen_optimizer.state_dict(),
                    'dis_optimizer': dis_optimizer.state_dict(),
                    'path_helper': args.path_helper,
                }, args.path_helper['ckpt_path'], filename="checkpoint_best_moment")
                is_best_moment = False
            elif is_best_cor:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'gen_model': args.gen_model,
                    'dis_model': args.dis_model,
                    'gen_state_dict': gen_net.state_dict(),
                    'dis_state_dict': dis_net.state_dict(),
                    'gen_optimizer': gen_optimizer.state_dict(),
                    'dis_optimizer': dis_optimizer.state_dict(),
                    'path_helper': args.path_helper,
                }, args.path_helper['ckpt_path'], filename="checkpoint_best_cor")
                is_best_cor = False
            elif is_best_p:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'gen_model': args.gen_model,
                    'dis_model': args.dis_model,
                    'gen_state_dict': gen_net.state_dict(),
                    'dis_state_dict': dis_net.state_dict(),
                    'gen_optimizer': gen_optimizer.state_dict(),
                    'dis_optimizer': dis_optimizer.state_dict(),
                    'path_helper': args.path_helper,
                }, args.path_helper['ckpt_path'], filename="checkpoint_best_p")
                is_best_p = False

    print('===============================================')
    print('Training Finished & Model Saved, the path is: ', args.path_helper['ckpt_path'] + '/' + 'checkpoint')

if __name__ == '__main__':
    main()
