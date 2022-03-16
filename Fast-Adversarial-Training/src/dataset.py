import os
import re
import sys
import json
import random
import numpy as np
from PIL import Image
import torch
import torch.utils.data
from omegaconf import OmegaConf
from torchvision import transforms
from iopath.common.file_io import PathManagerFactory
module_path = "../src"
if module_path not in sys.path:
    sys.path.append(module_path)
pathmgr = PathManagerFactory.get()


class Imagenet(torch.utils.data.Dataset):
    """ImageNet dataset."""

    def __init__(self, cfg, mode, num_retries=10):
        self.num_retries = num_retries
        self.cfg = cfg
        self.mode = mode
        self.data_path = cfg.data
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for ImageNet".format(mode)
        print("Constructing ImageNet {}...".format(mode))
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        split_path = os.path.join(self.data_path, self.mode)
        split_files = pathmgr.ls(split_path)
        self._class_ids = sorted(
            f for f in split_files if re.match(r"^n[0-9]+$", f)
        )
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)
            for im_name in pathmgr.ls(im_dir):
                im_path = os.path.join(im_dir, im_name)
                self._imdb.append({"im_path": im_path, "class": cont_id})
        print("Number of images: {}".format(len(self._imdb)))
        print("Number of classes: {}".format(len(self._class_ids)))


    def _prepare_im_tf(self, im_path):
        with pathmgr.open(im_path, "rb") as f:
            with Image.open(f) as im:
                im = im.convert("RGB")
        
        # Resize Step
        if self.cfg.AUGMENTATION.img_size > 0: 
            resize_transform = [ transforms.Resize(self.cfg.AUGMENTATION.img_size,
                                                       interpolation = 3)] 
        else:
            resize_transform = []
        
        # Rotation transform
        if self.cfg.AUGMENTATION.rot_inv:
            rotation_transform = [transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                                  transforms.RandomApply([transforms.RandomRotation((180, 180))], p=0.5)]
        else:
            rotation_transform = []
        
        # GrayScale transform
        if self.cfg.AUGMENTATION.grayscale:
            grayscale_transform = [transforms.RandomGrayscale(p=0.25)]
        else:
            grayscale_transform = []
            
        # Normalization Done Inside train_adv.py
        if self.mode == "train":
            
            aug_transform = transforms.Compose( resize_transform + rotation_transform + grayscale_transform + [
                                                transforms.RandomResizedCrop(self.cfg.AUGMENTATION.crop_size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()])
        else:
            aug_transform = transforms.Compose(resize_transform + [
                                               transforms.CenterCrop(self.cfg.AUGMENTATION.crop_size),
                                               transforms.ToTensor()])
        
        im = aug_transform(im)
        
        return im

    def __load__(self, index):
        # Load the image
        im_path = self._imdb[index]["im_path"]
        # Prepare the image for training / testing
        im = self._prepare_im_tf(im_path)
        return im

    def __getitem__(self, index):
        for _ in range(self.num_retries):
            im = self.__load__(index)
            if im is None:
                index = random.randint(0, len(self._imdb) - 1)
            else:
                break
                
        # Retrieve the label
        label = self._imdb[index]["class"]
        if isinstance(im, list):
            label = [label for _ in range(len(im))]
            return im, label
        else:
            return im, label

    def __len__(self):
        return len(self._imdb)

if __name__ == '__main__':
    cfg = {'AUGMENTATION' : {'img_size'  : 0,
                     'crop_size' : 224,
                     'rotation':False,
                     'grayscale':False,
                     'rot_inv':False},
           'data'     : "../../../BrainScore/Dataset/ILSVRC/Data/imagenet",
           'TRAIN': {'mean': [0.485, 0.456, 0.406],'std' : [0.229, 0.224, 0.225]},
           }
    cfg = OmegaConf.create(cfg)
    dataset = Imagenet(cfg,'train',num_retries = 10)
    print(dataset.__getitem__(0)[0].mean())
