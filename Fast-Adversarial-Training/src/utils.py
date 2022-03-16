import logging
import os
import random
import numpy as np
import torch
import datetime
import math
import torch
import yaml
from omegaconf import OmegaConf
import shutil

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(initial_lr, optimizer, epoch, n_repeats):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // int(math.ceil(30./n_repeats))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def initiate_logger(configs, evaluate):
    if not os.path.isdir(os.path.join(configs.PROJECT.output_dir,configs.PROJECT.runname,'ouput_logger')):
        os.makedirs(os.path.join(configs.PROJECT.output_dir,configs.PROJECT.runname,'ouput_logger'))
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(configs.PROJECT.output_dir,configs.PROJECT.runname,'ouput_logger', 'eval.txt' if evaluate else 'log.txt'),'w'))
    logger.info(pad_str(' LOGISTICS '))
    logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logger.info('Output Name: {}'.format(os.path.join(configs.PROJECT.output_dir,configs.PROJECT.runname,'ouput_logger')))
    logger.info('User: {}'.format(os.getenv('USER')))
    return logger

def pad_str(msg, total_len=70):
    rem_len = total_len - len(msg)
    return '*'*int(rem_len/2) + msg + '*'*int(rem_len/2)

def parse_config_file(args):
    with open(args.config) as f:
        config = OmegaConf.load(f)
    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v
    return config


def save_checkpoint(state, is_best, filepath, epoch):
    filename = os.path.join(filepath, f'checkpoint_epoch{epoch}.pth.tar')
    # Save model
    torch.save(state, filename)
    # Save best model
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))


def seed_everything(seed=42):
    '''
    Function to put a seed to every step and make code reproducible
    Input:
    - seed: random state for the events 
    '''
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
