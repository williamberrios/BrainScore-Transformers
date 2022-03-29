import sys
import os
import functools
import torch
import pandas as pd
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.check_submission import check_models
from model_tools.brain_transformation import ModelCommitment
from timm.models import create_model
from model_tools.activations.pytorch import load_images
import numpy as np
from brainscore import score_model
import argparse
from omegaconf import OmegaConf
import boto3
import json
from decimal import Decimal
from settings import keys


# +
def put_device(keys,item):
    dynamodb = boto3.resource('dynamodb',
                          region_name='us-east-1',
                          aws_access_key_id     = keys.AWS_SERVER_PUBLIC_KEY,
                          aws_secret_access_key = keys.AWS_SERVER_SECRET_KEY)
    table = dynamodb.Table(keys.TABLE)
    response = table.put_item(Item=item)

    
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config',type=str,default='config.yaml' ,help='path to dataset')
    return parser.parse_args()

def parse_config_file(args):
    with open(args.config) as f:
        config = OmegaConf.load(f)
    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v
    return config


# +
def load_checkpoint(ckpt,mode ='FAT'):
    if mode == 'FAT':
        checkpoint = torch.load(ckpt)
        checkpoint_new = torch.load(ckpt)
        for key in checkpoint['state_dict'].keys():
            checkpoint_new['state_dict'][key[7:]] = checkpoint['state_dict'][key]
            del checkpoint_new['state_dict'][key]
        del checkpoint
        return checkpoint_new
    elif mode == 'normal':
        return torch.load(ckpt)
    
def get_layers(model):
    layers = []
    for name, layer in model.named_modules():
        if len(list(layer.children())) > 0:  # this module only holds other modules
            continue
        if name != "":
            layers.append(str(name))

    return layers


# +
def preprocess_images(images, image_size, **kwargs):
    preprocess = torchvision_preprocess_input(image_size, **kwargs)
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images


def torchvision_preprocess_input(image_size, **kwargs):
    from torchvision import transforms
    return transforms.Compose([
        # Modify the load
        transforms.Resize((256, 256)),
        transforms.CenterCrop((image_size,image_size)),
        torchvision_preprocess(**kwargs),
    ])

def torchvision_preprocess(normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)):
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])

def custom_load_preprocess_images(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    return images


# -

def main():
    #configs = OmegaConf.load("Configs/ViT-Base-Patch-16-224-FAT-ROTINV.yaml")
    configs = parse_config_file(parse_args())
    model = create_model(configs.MODEL.arch,pretrained = configs.MODEL.pretrained)
    if os.path.isfile(configs.MODEL.checkpoint):
        print(f"Loading Checkpoint {configs.MODEL.checkpoint} ...")
        checkpoint = load_checkpoint(configs.MODEL.checkpoint,configs.MODEL.mode)
        model.load_state_dict(checkpoint['state_dict'],strict = True)

    else:
        if configs.MODEL.checkpoint=="":
            pass
        else:
            raise Exception(f"=> no checkpoint found at {configs.MODEL.checkpoint}")

    model.eval()

    if configs.DATASET_LAYERS_DICT is None:
        LAYERS = get_layers(model)
        DATASETS = ['movshon.FreemanZiemba2013public.V1-pls',
                    'movshon.FreemanZiemba2013public.V2-pls',
                    'dicarlo.MajajHong2015public.V4-pls',
                    'dicarlo.MajajHong2015public.IT-pls',
                    'dicarlo.Rajalingham2018public-i2n']

        DATASET_LAYERS_DICT = {}
        for dataset in DATASETS:
            DATASET_LAYERS_DICT[dataset] = LAYERS
    else:
        DATASET_LAYERS_DICT = config.DATASET_LAYERS_DICT
        
    for dataset,layers in DATASET_LAYERS_DICT.keys():
        print(f"{'*'*10} Dataset: {dataset} {'*'*10}")
        for layer in layers:
            print(f"{'*'*10} Layer: {layer} {'*'*10}")
            preprocessing = functools.partial(custom_load_preprocess_images, 
                                              image_size = CROP_SIZE)
            activations_model = PytorchWrapper(identifier="tmp", 
                                               model=model, 
                                               preprocessing= preprocessing,
                                               batch_size = BATCH_SIZE)

            brain_model       = ModelCommitment(identifier="tmp", 
                                                activations_model=activations_model,
                                                layers=[layer])

            score = score_model(model_identifier = brain_model.identifier, 
                                model = brain_model,
                                benchmark_identifier = DATASET)
            center, error = score.sel(aggregation='center'), score.sel(aggregation='error')
            dict_ = {"id": 'Model:' + configs.MODEL.arch + ' Config:' + configs.config.split('/')[-1]
                     "id_join": configs.REF.run,
                     "dataset": dataset,
                     "layer"  : layer,
                     "score-center":center,
                     "score-error" :error}
            print(dict_)
            dict_ = json.loads(json.dumps(dict_), parse_float=Decimal)
            # Update in database
            put_device(keys,dict_)


if __name__ == '__main__':
    main()

'''
MODEL_NAME      = "vit_base_patch16_224"
PRETRAINED      = True
CHECKPOINT_NAME = "../Fast-Adversarial-Training/SavedModels/ViT-Base-Patch-16-FAT-ROTINV/trained_models/checkpoint_epoch5.pth.tar"
MODE = 'FAT'
BATCH_SIZE = 1
RESIZE_SIZE = 256
#REF.run?
CROP_SIZE   = 224
'''
