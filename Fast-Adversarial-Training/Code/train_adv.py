import os
import sys
import math
import time
import wandb
import argparse
import torch
import torch.optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torchvision.datasets as datasets
from torch.autograd import Variable
from torchvision import transforms
from torch.cuda.amp import GradScaler
from timm.models import create_model
from omegaconf import OmegaConf
module_path = "../src"
if module_path not in sys.path:
    sys.path.append(module_path)
from dataset import Imagenet
from scheduler import custom_lr_scheduler_v1
from utils import *
from validation import validate, validate_pgd
import warnings
warnings.filterwarnings("ignore")
os.environ['WANDB_SILENT']="true"


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data',type=str,default='Dataset/imagenet' ,help='path to dataset')
    parser.add_argument('--config', default='configs.yml', type=str, metavar='Path',help='path to the config file (default: configs.yml)')
    parser.add_argument('--resume', default='', type=str,help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')    
    return parser.parse_args()


def main():
    # Make Reproducible code:
    seed_everything(configs.TRAIN.seed)
    # Initialize wandb
    if configs.PROJECT.wandb:
        run = wandb.init(project = configs.PROJECT.project_name,
                         save_code = True,
                         reinit    = True)
        run.name = configs.PROJECT.runname
        run.save()
    else:
        run = None    
    # Scale and initialize the parameters
    best_prec1 = 0
    configs.TRAIN.epochs = int(math.ceil(configs.TRAIN.epochs))
    configs.ADV.fgsm_step /= configs.DATA.max_color_value
    configs.ADV.clip_eps /= configs.DATA.max_color_value
    
    # Create output folder
    if not os.path.isdir(os.path.join(configs.PROJECT.output_dir,configs.PROJECT.runname,'trained_models')):
        os.makedirs(os.path.join(configs.PROJECT.output_dir,configs.PROJECT.runname,'trained_models'))
    
    # Log the config details
    logger.info(pad_str(' ARGUMENTS '))
    logger.info(pad_str(''))
    print(OmegaConf.to_yaml(configs))

    
    # Create the model
    model = create_model(configs.TRAIN.arch,pretrained = configs.TRAIN.pretrained_arch)  
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()       
    
    if configs.TRAIN.optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.TRAIN.lr, weight_decay=configs.TRAIN.weight_decay)
    else:
        raise Exception("Optimizer not Implemented yet")

    model = torch.nn.DataParallel(model)

    if configs.resume:
        if os.path.isfile(configs.resume):
            print(f"=> loading checkpoint {configs.resume}")
            checkpoint = torch.load(configs.resume)
            configs.TRAIN.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint: {configs.resume} (epoch {checkpoint['epoch']})")
            del checkpoint
            torch.cuda.empty_cache()
        else:
            raise Exception(f"=> no checkpoint found at {configs.resume}")
    train_dataset = Imagenet(configs,'train')    
    val_dataset = Imagenet(configs,'val')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size  = configs.DATA.batch_size, 
                                               shuffle     = True,
                                               num_workers = configs.DATA.workers, 
                                               pin_memory  = True, 
                                               sampler     = None)
    

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size  = configs.DATA.batch_size, 
                                             shuffle     = False,
                                             num_workers = configs.DATA.workers, 
                                             pin_memory  = True)
    total_steps = len(train_loader)
    if configs.evaluate:
        logger.info(pad_str(' Performing PGD Attacks '))
        for pgd_param in configs.ADV.pgd_attack:
            validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs, logger,configs.TRAIN.arch)
        validate(val_loader, model, criterion, configs, logger,configs.TRAIN.arch)
        return
    
    print(" ************* Init Training ********************* ")
    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        # train for one epoch
        print(f"=================== Epoch: {epoch} ==================")
        train(train_loader, model, criterion, optimizer, epoch, configs.TRAIN.half,run)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, configs, logger,configs.TRAIN.arch)            
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': configs.TRAIN.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, os.path.join(configs.PROJECT.output_dir,configs.PROJECT.runname,'trained_models'),epoch + 1)
        if run is not None:
            run.log({'Valid_Prec@1_epoch':np.round(prec1.cpu().numpy(),4)})
            run.log({'Best_Prec@1_epoch':np.round(best_prec1.cpu().numpy(),4)})
    if run is not None:
        run.finish()

global global_noise_data
global_noise_data = torch.zeros([configs.DATA.batch_size, 3, configs.AUGMENTATION.crop_size, configs.AUGMENTATION.crop_size]).cuda()
def train(train_loader, model, criterion, optimizer, epoch, half=False,run = None): 
    global global_noise_data
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,configs.AUGMENTATION.crop_size, configs.AUGMENTATION.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.AUGMENTATION.crop_size, configs.AUGMENTATION.crop_size).cuda()

    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    if half:
        scaler1 = GradScaler()
        scaler2 = GradScaler()
    for i, (input, target) in (enumerate(train_loader)):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)
        if getattr(configs.ADV,'enable',True):
            if configs.TRAIN.random_init: 
                global_noise_data.uniform_(-configs.ADV.clip_eps, configs.ADV.clip_eps)
        
        if configs.TRAIN.scheduler_name == 'custom_lr_scheduler_v1':
            lr = custom_lr_scheduler_v1(epoch,i,len(train_loader))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif configs.TRAIN.scheduler_name == 'None':
            pass
        else:
            raise Exception('Choose a Valid Scheduler')
        
        if configs.ADV.enable:
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True)
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)

            if half:
                with torch.cuda.amp.autocast():
                    output = model(in1)
                    loss = criterion(output, target)
                scaler1.scale(loss).backward()
            else:
                output = model(in1)
                loss = criterion(output, target)
                loss.backward()    
                
            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-configs.ADV.clip_eps, configs.ADV.clip_eps)

            # Descend on global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=False)
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)
            
        else:
            # Normalize batch of images
            in1 = input
            in1.sub_(mean).div_(std)
        
        optimizer.zero_grad()
        
        if half: 
            with torch.cuda.amp.autocast():
                output = model(in1)
                loss = criterion(output, target)
            scaler2.scale(loss).backward()
            scaler2.step(optimizer)
            scaler2.update()
        else:
            output = model(in1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % configs.TRAIN.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})\t'
                  'LR {lr:.6f}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, top1=top1,
                   top5=top5,cls_loss=losses, lr=lr))

            sys.stdout.flush()
            if run is not None:
                run.log({'lr':optimizer.param_groups[0]['lr']})
                run.log({'Train_Prec@1_avg':np.round(top1.avg.cpu().numpy(),3)})
                run.log({'Train_Prec@5_avg':np.round(top5.avg.cpu().numpy(),3)})


if __name__ == '__main__':
    # Start Config and Logger
    configs = parse_config_file(parse_args())
    logger = initiate_logger(configs,configs.evaluate)
    main()
