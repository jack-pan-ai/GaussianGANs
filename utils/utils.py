import logging
import os
import time
from datetime import datetime
import dateutil.tz
import torch

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # image save
    prefix_image = os.path.join(prefix, 'image')

    log_path_img_pca = os.path.join(prefix_image, 'pca')
    os.makedirs(log_path_img_pca)
    path_dict['log_path_img_pca'] = log_path_img_pca

    log_path_img_heatmap = os.path.join(prefix_image, 'heatmap')
    os.makedirs(log_path_img_heatmap)
    path_dict['log_path_img_heatmap'] = log_path_img_heatmap

    log_path_img_qqplot = os.path.join(prefix_image, 'qqplot')
    os.makedirs(log_path_img_qqplot)
    path_dict['log_path_img_qqplot'] = log_path_img_qqplot

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict

def save_checkpoint(states, output_dir, filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))

def load_checkpoint(net, checkpoint_PATH, optimizer=None):
    cwd = os.getcwd()
    CKPT_PATH = os.path.join(cwd, checkpoint_PATH)
    # print('Loaded Path: ', CKPT_PATH)
    if CKPT_PATH is not None:
        print('loading checkpoint!')
        CKPT = torch.load(CKPT_PATH)
        net.load_state_dict(CKPT['gen_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(CKPT['gen_optimizer'])
    else:
        print('Please input the model path')



