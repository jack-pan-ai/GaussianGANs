from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import cfg
from dataset.UniMiB.dataLoader import *
from models.GANModels import *
from utils.functions import train, save_samples, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger
from utils.visualizationMetrics import visualization
from utils.statEstimate import *
from utils.utils import load_checkpoint
from models.inverseGANModels import inverseGenerator

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils import data
import os
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import random
import matplotlib.pyplot as plt
from torchinfo import summary
from einops import rearrange


# synthesis data
import warnings
import wandb

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True



def main():
    args = cfg.parse_args()
    
    if args.seed is not None:
        # reproduce the results
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False # the accelerator for CNN computation
        torch.backends.cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
         args.world_size = int(os.environ["WORLD_SIZE"])#

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
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
#         elif classname.find('Linear') != -1:
#             if args.init_type == 'normal':
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
#             elif args.init_type == 'orth':
#                 nn.init.orthogonal_(m.weight.data)
#             elif args.init_type == 'xavier_uniform':
#                 nn.init.xavier_uniform(m.weight.data, 1.)
#             else:
#                 raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    # dataset loading
    print('------------------------------------------------------------')
    print('Data Loadering ~ Wait ~')
    print('------------------------------------------------------------')

    # train_set = VTSDataset(set_type = 'train')
    # test_set = TSDataset(set_type = 'test')
    # [batch_size, channles, seq-len]
    train_set = unimib_load_dataset(incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=False, is_normalize=True,
                                    one_hot_encode=False, data_mode='Train', single_class=True,
                                    class_name=args.class_name, augment_times=args.augment_times)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_set = unimib_load_dataset(incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=False, is_normalize=True,
                                   one_hot_encode=False, data_mode='Test', single_class=True,
                                   class_name=args.class_name)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print('------------------------------------------------------------')
    print('How many iterations in a training epoch: ', len(train_loader))
    print('------------------------------------------------------------')
    assert len(train_set) >= args.eval_num, 'The number of evaluation should be less than test_set'

    channels, seq_len = train_set[1].shape
    # import network
    gen_net = Generator(seq_len=seq_len, channels=channels,
                        num_heads= args.heads, latent_dim=args.noise_dim,
                        depth=args.g_depth, patch_size=args.patch_size)
    dis_net = Discriminator(seq_len=seq_len, channels=channels,
                            num_heads=args.heads, depth=args.d_depth,
                            patch_size=args.patch_size)
    # import inverse network
    if args.checkpoint_best_PATH is not None:
        gen_inv_net = inverseGenerator(seq_len=seq_len, channels=channels,
                            num_heads= args.heads, latent_dim=args.noise_dim,
                            depth=args.g_depth, patch_size=args.patch_size).cuda(0)
        checkpoint_best_PATH = args.checkpoint_best_PATH
        load_checkpoint(gen_inv_net, checkpoint_best_PATH)
        gen_inv_net.eval()

    #print(dis_net)
    print('------------------------------------------------------------')
    print('Copy Gen/Dis networks to gpu')
    print('------------------------------------------------------------')
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
#             gen_net = eval('models_search.'+args.gen_model+'.Generator')(args=args)
#             dis_net = eval('models_search.'+args.dis_model+'.Discriminator')(args=args)

            gen_net.apply(weights_init)
            dis_net.apply(weights_init)
            gen_net.cuda(args.gpu)
            dis_net.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.dis_batch_size = int(args.dis_batch_size / ngpus_per_node)
            args.gen_batch_size = int(args.gen_batch_size / ngpus_per_node)
            args.batch_size = args.dis_batch_size
            
            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            gen_net = torch.nn.parallel.DistributedDataParallel(gen_net, device_ids=[args.gpu], find_unused_parameters=True)
            dis_net = torch.nn.parallel.DistributedDataParallel(dis_net, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            gen_net.cuda()
            dis_net.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            gen_net = torch.nn.parallel.DistributedDataParallel(gen_net)
            dis_net = torch.nn.parallel.DistributedDataParallel(dis_net)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        gen_net.cuda(args.gpu)
        dis_net.cuda(args.gpu)
    else:
        gen_net = torch.nn.DataParallel(gen_net).cuda()
        dis_net = torch.nn.DataParallel(dis_net).cuda()
    print('------------------------------------------------------------')
    print('The gpu on your deivce: cuda', np.arange(ngpus_per_node))
    if args.gpu is not None:
        print('However, you are just using gpu', args.gpu)
    print('------------------------------------------------------------')

    # model size for generator
    summary(gen_net, (args.batch_size, args.noise_dim))

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

    # initial
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    start_epoch = 0
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_num, args.noise_dim)))

    # set writer
    writer = None
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']

        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(gen_net, mode='gpu')
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        fixed_z = checkpoint['fixed_z']

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path']) if args.rank == 0 else None
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        writer = SummaryWriter(args.path_helper['log_path']) if args.rank == 0 else None
        del checkpoint
    else:
    # create new log dir
        assert args.exp_name
        if args.rank == 0:
            args.path_helper = set_log_dir('logs', args.exp_name)
            logger = create_logger(args.path_helper['log_path'])
            writer = SummaryWriter(args.path_helper['log_path'])
    
    if args.rank == 0:
        logger.info(args)
    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }
    print('------------------------------------------------------------')
    print(f"Log file path: {args.path_helper['prefix']}") if args.rank == 0 else 0
    print('------------------------------------------------------------')

    # wandb ai monitoring
    project_name = 'loss: ' + args.loss + ', n_gen: ' + str(args.n_gen) + ', n_dis: '+ str(args.n_dis)
    wandb.init(project=args.exp_name, entity="qilong77", name = project_name)
    wandb.config = {
        "epochs": int(args.epochs) - int(start_epoch),
        "batch_size": args.batch_size
    }

    # train loop
    for epoch in range(int(start_epoch), int(args.epochs)):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param,
              train_loader, epoch, writer_dict, lr_schedulers)

        # save the generated time series after using PCA and t-SNE

        if args.rank == 0 or args.show:
            if (epoch) % args.eval_epochs == 0:
                #backup_param = copy_params(gen_net)
                #load_params(gen_net, gen_avg_param, args, mode="cpu")
                # load_params(gen_net, backup_param, args)
                with torch.no_grad():
                    # sample_imgs is on GPU device
                    sample_imgs = gen_net(fixed_z).detach()
                save_samples(args, fixed_z[0].reshape(1, -1), epoch + 1, gen_net, writer_dict)
                # evaluation the net from different perspective
                visualization(ori_data=train_set[:args.eval_num], generated_data=sample_imgs.to('cpu'), analysis='pca',
                              save_name=args.exp_name, epoch=epoch, args=args)
                with torch.no_grad():
                    sample_trans_imgs = gen_inv_net(sample_imgs.to('cuda:0')).detach().to('cpu')
                qqplot(sample_trans_imgs.squeeze(1), epoch=epoch, args=args)
                heatmap_cor(sample_trans_imgs.squeeze(1), epoch=epoch, args=args)
                if epoch == 0:
                    is_best_det = False
                    det_dis = diff_cor(sample_trans_imgs.squeeze(1))
                    det_dis_best = det_dis
                else:
                    det_dis = diff_cor(sample_trans_imgs.squeeze(1))
                    if det_dis < det_dis_best:
                        det_dis_best = det_dis
                        is_best_det = True
                    else:
                        is_best_det = False
                visu_pca = plt.imread(
                    os.path.join(args.path_helper['log_path_img_pca'], f'{args.exp_name}_epoch_{epoch + 1}.png'))
                visu_qqplot = plt.imread(
                    os.path.join(args.path_helper['log_path_img_qqplot'], f'{args.exp_name}_epoch_{epoch + 1}.png'))
                visu_heatmap = plt.imread(
                    os.path.join(args.path_helper['log_path_img_heatmap'], f'{args.exp_name}_epoch_{epoch + 1}.png'))
                img_visu_pca = wandb.Image(visu_pca, caption="Epoch: " + str(epoch))
                img_visu_qqplot = wandb.Image(visu_qqplot, caption="Epoch: " + str(epoch))
                img_visu_heatmap = wandb.Image(visu_heatmap, caption="Epoch: " + str(epoch))
                wandb.log({'PCA': img_visu_pca,
                           'QQplot': img_visu_qqplot,
                           'Heatmap': img_visu_heatmap})
                wandb.log({'Determinant': det_dis})
                # image = plt.imread(os.path.join(args.path_helper['log_path_img'],f'{args.exp_name}_epoch_{epoch+1}.png'))
                # wandb.log({'PCA plot': image})
                # image = rearrange(image, 'h w c -> c h w')
                # writer.add_image('Comparison for original and generative data based on PCA', image, epoch + 1)

        # save the model
        if epoch != 0:
            if is_best_det:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'gen_model': args.gen_model,
                    'dis_model': args.dis_model,
                    'gen_state_dict': gen_net.state_dict(),
                    'dis_state_dict': dis_net.state_dict(),
                    'gen_optimizer': gen_optimizer.state_dict(),
                    'dis_optimizer': dis_optimizer.state_dict(),
                    'path_helper': args.path_helper,
                }, args.path_helper['ckpt_path'], filename="checkpoint_best_det")

        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param, args)
        save_checkpoint({
            'epoch': epoch + 1,
            'gen_model': args.gen_model,
            'dis_model': args.dis_model,
            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'avg_gen_state_dict': avg_gen_net.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'path_helper': args.path_helper,
        }, args.path_helper['ckpt_path'], filename="checkpoint")
        del avg_gen_net

    print('===============================================')
    print('Training Finished & Model Saved, the path is: ', args.path_helper['ckpt_path']+ '/'+ 'checkpoint')

if __name__ == '__main__':
    main()
