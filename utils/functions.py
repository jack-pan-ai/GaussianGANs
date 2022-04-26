# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Version : 0.0
# @Link    : None
## @ revised : 2022-04-07

## @ Author: Qilong Pan (qilong.pan@kaust.edu.sa)
## @ Version: 0.0
## @ link: jack-pan-ai.github.com

import logging
import operator
import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle

import wandb


logger = logging.getLogger(__name__)

def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty

def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
          epoch, writer_dict, schedulers=None):
    writer = writer_dict['writer'] # it would influence the process of debug
    gen_step = 0
    # train mode
    gen_net.train()
    dis_net.train()
    
    dis_optimizer.zero_grad() # here not wrong, it's a wise poistion of zero gradients
    gen_optimizer.zero_grad() # accumualted_times would give the zero gradients in the the bacth iteration
    g_loss_sum = 0
    d_loss_sum = 0
    for iter_idx, imgs in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']
        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)
        # Sample noise as generator input [batch_size, noise_dim] (noise _dim is dimension for Gaussain Noise)
        z = torch.randn([imgs.shape[0], args.noise_dim]).cuda(args.gpu, non_blocking=True)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(args.n_dis):
            real_validity = dis_net(real_imgs)
            fake_imgs = gen_net(z).detach()
            assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"
            fake_validity = dis_net(fake_imgs)

            if args.loss == 'standard':
                #soft label
                real_label = torch.full((real_validity.shape[0],), 0.9, dtype=torch.float, device=real_imgs.get_device())
                fake_label = torch.full((fake_validity.shape[0],), 0.1, dtype=torch.float, device=real_imgs.get_device())
                real_validity = real_validity.view(-1)
                fake_validity = fake_validity.view(-1)
                d_real_loss = nn.BCELoss()(real_validity, real_label)
                d_fake_loss = nn.BCELoss()(fake_validity, fake_label)
                d_loss = d_real_loss + d_fake_loss
            elif args.loss == 'lsgan':
                real_label = torch.full(real_validity.shape, 1., dtype=torch.float, device=real_imgs.get_device())
                fake_label = torch.full(real_validity.shape, 0., dtype=torch.float, device=real_imgs.get_device())
                d_real_loss = nn.MSELoss()(real_validity, real_label)
                d_fake_loss = nn.MSELoss()(fake_validity, fake_label)
                d_loss = d_real_loss + d_fake_loss
            elif args.loss == 'wgangp':
                gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                        args.phi ** 2)
            elif args.loss == 'wgangp-eps':
                gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                        args.phi ** 2)
                d_loss += (torch.mean(real_validity) ** 2) * 1e-3
            elif args.loss == 'wgangp-mode':
                gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                        args.phi ** 2)
            else:
                raise NotImplementedError(args.loss)
            d_loss = d_loss/float(args.accumulated_times)
            d_loss.backward()

        # visulize the training process for loss and validity
        if (iter_idx + 1) % args.accumulated_times == 0:
            torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
            dis_optimizer.step()
            dis_optimizer.zero_grad()
            writer.add_scalar('d_loss', d_loss.item(), global_steps) if args.rank == 0 else 0
            writer.add_scalar('Discriminator real validity', real_validity.mean().item(), global_steps) if args.rank == 0 else 0
            writer.add_scalar('Discriminator fake validity', fake_validity.mean().item(), global_steps) if args.rank == 0 else 0
            wandb.log({
                "Discriminator Real validity": real_validity.mean().item(),
                "Discriminator Fake validity": fake_validity.mean().item(),
                "Discriminator Loss per iteration": d_loss.item()})

        # -----------------
        #  Train Generator   (here it's wise coding way: the generator would be trained only when it's integer multiples of n_dis * accumualated_times)
        # -----------------
        if global_steps % args.accumulated_times == 0:
            for _ in range(args.n_gen):
                for accumulated_idx in range(args.g_accumulated_times):
                    gen_z =  torch.randn([imgs.shape[0], args.noise_dim]).cuda(args.gpu, non_blocking = True)
                    gen_imgs = gen_net(gen_z)
                    fake_validity = dis_net(gen_imgs)

                    # cal loss
                    loss_lz = torch.tensor(0)
                    if args.loss == "standard":
                        # -log(D(G(z)))
                        fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                        g_loss = - torch.sum(torch.log(fake_validity))
                    elif args.loss == "lsgan":
                        real_label = torch.full((fake_validity.shape[0],), 1., dtype=torch.float, device=real_imgs.get_device())
                        g_loss = nn.MSELoss()(fake_validity.view(-1), real_label)
                    elif args.loss == 'wgangp-mode':
                        fake_image1, fake_image2 = gen_imgs[:args.batch_size//2], gen_imgs[args.batch_size//2:]
                        z_random1, z_random2 = gen_z[:args.batch_size//2], gen_z[args.batch_size//2:]
                        lz = torch.mean(torch.abs(fake_image2 - fake_image1)) / torch.mean(
                        torch.abs(z_random2 - z_random1))
                        eps = 1 * 1e-5
                        loss_lz = 1 / (lz + eps)

                        g_loss = -torch.mean(fake_validity) + loss_lz
                    else:
                        g_loss = -torch.mean(fake_validity)
                    g_loss = g_loss/float(args.g_accumulated_times)
                    g_loss.backward()
                    '''
                    TO DO: more loss functions
                    '''

                torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
                gen_optimizer.step()
                gen_optimizer.zero_grad()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            ema_nimg = args.ema_kimg * 1000
            cur_nimg = args.batch_size * args.world_size * global_steps
            if args.ema_warmup != 0:
                ema_nimg = min(ema_nimg, cur_nimg * args.ema_warmup)
                ema_beta = 0.5 ** (float(args.batch_size * args.world_size) / max(ema_nimg, 1e-8))
            else:
                ema_beta = args.ema
                
            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                cpu_p = deepcopy(p)
                avg_p.mul_(ema_beta).add_(1. - ema_beta, cpu_p.cpu().data)
                del cpu_p

            writer.add_scalar('g_loss', g_loss.item(), global_steps//args.n_dis) if args.rank == 0 else 0
            gen_step += 1
            wandb.log({"Generator loss per iteration": g_loss.item()})

        # verbose
        if gen_step and iter_idx % args.print_freq == 0 and args.rank == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [ema: %f] " %
                (epoch, args.epochs, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(), ema_beta))
            del gen_imgs
            del real_imgs
            del fake_validity
            del real_validity
            del g_loss
            del d_loss

        writer_dict['train_global_steps'] = global_steps + 1


def inverse_train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
          epoch, writer_dict, img_set, schedulers=None):
    writer = writer_dict['writer']  # it would influence the process of debug
    gen_step = 0
    # train mode
    gen_net.train()
    dis_net.train()

    dis_optimizer.zero_grad()  # here not wrong, it's a wise poistion of zero gradients
    gen_optimizer.zero_grad()  # accumualted_times would give the zero gradients in the the bacth iteration
    g_loss_sum = 0
    d_loss_sum = 0
    for iter_idx, imgs in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']
        # Adversarial ground truths
        real_z = imgs.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)
        # Sample noise as generator input [batch_size, noise_dim] (noise _dim is dimension for Gaussain Noise)
        shuffle_no = torch.randint(0, len(img_set), [real_z.shape[0]])
        imgs = torch.from_numpy(img_set[shuffle_no]).type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        real_validity = dis_net(real_z)
        fake_z = gen_net(imgs).detach()
        assert fake_z.size() == real_z.size(), f"fake_z.size(): {fake_z.size()} real_z.size(): {real_z.size()}"
        fake_validity = dis_net(fake_z)

        if args.loss == 'standard':
            # soft label
            real_label = torch.full((real_validity.shape[0],), 0.9, dtype=torch.float, device=real_z.get_device())
            fake_label = torch.full((fake_validity.shape[0],), 0.1, dtype=torch.float, device=real_z.get_device())
            real_validity = real_validity.view(-1)
            fake_validity = fake_validity.view(-1)
            d_real_loss = nn.BCELoss()(real_validity, real_label)
            d_fake_loss = nn.BCELoss()(fake_validity, fake_label)
            d_loss = d_real_loss + d_fake_loss
        elif args.loss == 'lsgan':
            real_label = torch.full(real_validity.shape, 1., dtype=torch.float, device=real_z.get_device())
            fake_label = torch.full(real_validity.shape, 0., dtype=torch.float, device=real_z.get_device())
            d_real_loss = nn.MSELoss()(real_validity, real_label)
            d_fake_loss = nn.MSELoss()(fake_validity, fake_label)
            d_loss = d_real_loss + d_fake_loss
        elif args.loss == 'wgangp':
            gradient_penalty = compute_gradient_penalty(dis_net, real_z, fake_z.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
        elif args.loss == 'wgangp-eps':
            gradient_penalty = compute_gradient_penalty(dis_net, real_z, fake_z.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
            d_loss += (torch.mean(real_validity) ** 2) * 1e-3
        elif args.loss == 'wgangp-mode':
            gradient_penalty = compute_gradient_penalty(dis_net, real_z, fake_z.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
        else:
            raise NotImplementedError(args.loss)
        d_loss = d_loss / float(args.accumulated_times)
        d_loss.backward()

        # visulize the training process for loss and validity
        if (iter_idx + 1) % args.accumulated_times == 0:
            torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
            dis_optimizer.step()
            dis_optimizer.zero_grad()
            writer.add_scalar('d_loss', d_loss.item(), global_steps) if args.rank == 0 else 0
            writer.add_scalar('Discriminator real validity', real_validity.mean().item(),
                              global_steps) if args.rank == 0 else 0
            writer.add_scalar('Discriminator fake validity', fake_validity.mean().item(),
                              global_steps) if args.rank == 0 else 0
            wandb.log({
                "Discriminator Real validity": real_validity.mean().item(),
                "Discriminator Fake validity": fake_validity.mean().item(),
                "Discriminator Loss per iteration": d_loss.item()})

        # -----------------
        #  Train Generator   (here it's wise coding way: the generator would be trained only when it's integer multiples of n_dis * accumualated_times)
        # -----------------
        if global_steps % (args.n_dis * args.accumulated_times) == 0:

            for accumulated_idx in range(args.g_accumulated_times):
                imgs_n = torch.from_numpy(img_set[shuffle_no]).type(torch.cuda.FloatTensor).cuda(args.gpu,
                                                                                               non_blocking=True)
                imgs_n = gen_net(imgs_n)
                fake_validity = dis_net(imgs_n)

                # cal loss
                loss_lz = torch.tensor(0)
                if args.loss == "standard":
                    # -log(D(G(z)))
                    fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                    g_loss = - torch.sum(torch.log(fake_validity))
                elif args.loss == "lsgan":
                    real_label = torch.full((fake_validity.shape[0],), 1., dtype=torch.float,
                                            device=real_z.get_device())
                    g_loss = nn.MSELoss()(fake_validity.view(-1), real_label)
                elif args.loss == 'wgangp-mode':
                    fake_image1, fake_image2 = imgs_n[:args.batch_size // 2], imgs_n[args.batch_size // 2:]
                    z_random1, z_random2 = imgs_n[:args.batch_size // 2], imgs_n[args.batch_size // 2:]
                    lz = torch.mean(torch.abs(fake_image2 - fake_image1)) / torch.mean(
                        torch.abs(z_random2 - z_random1))
                    eps = 1 * 1e-5
                    loss_lz = 1 / (lz + eps)

                    g_loss = -torch.mean(fake_validity) + loss_lz
                else:
                    g_loss = -torch.mean(fake_validity)
                g_loss = g_loss / float(args.g_accumulated_times)
                g_loss.backward()
                '''
                TO DO: more loss functions
                '''

            torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
            gen_optimizer.step()
            gen_optimizer.zero_grad()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            ema_nimg = args.ema_kimg * 1000
            cur_nimg = args.batch_size * args.world_size * global_steps
            if args.ema_warmup != 0:
                ema_nimg = min(ema_nimg, cur_nimg * args.ema_warmup)
                ema_beta = 0.5 ** (float(args.batch_size * args.world_size) / max(ema_nimg, 1e-8))
            else:
                ema_beta = args.ema

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                cpu_p = deepcopy(p)
                avg_p.mul_(ema_beta).add_(1. - ema_beta, cpu_p.cpu().data)
                del cpu_p

            writer.add_scalar('g_loss', g_loss.item(), global_steps // args.n_dis) if args.rank == 0 else 0
            gen_step += 1
            wandb.log({"Generator loss per iteration": g_loss.item()})
            g_loss_sum = g_loss_sum + g_loss

        d_loss_sum = d_loss_sum + d_loss
        # verbose
        if gen_step and iter_idx % args.print_freq == 0 and args.rank == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [ema: %f] " %
                (epoch, args.epochs, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(),
                 ema_beta))
            del imgs_n
            del real_z
            del fake_validity
            del real_validity
            del g_loss
            del d_loss

        writer_dict['train_global_steps'] = global_steps + 1

def save_samples(args, fixed_z, epoch, gen_net: nn.Module, writer_dict, clean_dir=True):

    # eval mode
    gen_net.eval()
    with torch.no_grad():
        # generate images
        sample_imgs = []
        for i in range(fixed_z.size(0)):
            sample_img = gen_net(fixed_z[i])
            sample_imgs.append(sample_img)
        sample_imgs = torch.cat(sample_imgs, dim=0)
        os.makedirs(args.path_helper['sample_path'], exist_ok=True)
        with open(os.path.join(args.path_helper['sample_path'], f'sampled_time_series_epoch_{epoch}.pkl'), 'wb')as f:
            pickle.dump(sample_imgs, f)
        f.close()
    return sample_imgs

class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):
        '''
        If the dataset is not that large, this linear learning rate decay would not work very effectively.
        '''
        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr

def load_params(model, new_param, args, mode="gpu"):
    if mode == "cpu":
        for p, new_p in zip(model.parameters(), new_param):
            cpu_p = deepcopy(new_p)
#             p.data.copy_(cpu_p.cuda().to(f"cuda:{args.gpu}"))
            p.data.copy_(cpu_p.cuda().to("cpu"))
            del cpu_p
    
    else:
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)

def copy_params(model, mode='cpu'):
    if mode == 'gpu':
        flatten = []
        for p in model.parameters():
            cpu_p = deepcopy(p).cpu()
            flatten.append(cpu_p.data)
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten