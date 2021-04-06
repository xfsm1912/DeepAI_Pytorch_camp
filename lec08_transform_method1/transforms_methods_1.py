# -*- coding: utf-8 -*-
"""
# @file name  : transforms_methods_1.py
# @author     : Jianhua Ma
# @date       : 20210330
# @brief      : transforms method 1
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
import sys
from model.lenet import LeNet
from tools.my_dataset import RMBDataset
from tools.common_tools import set_seed, transform_invert

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

path_lenet = os.path.join(BASE_DIR, "..", "model", "lenet.py")
path_tools = os.path.join(BASE_DIR, "..", "tools", "common_tools.py")

assert os.path.exists(path_lenet), f"{path_lenet} not exist, please place lenet.py in the {os.path.dirname(path_lenet)}"
assert os.path.exists(path_tools), f"{path_tools} not exist, please place common_tools.py in the {os.path.dirname(path_tools)}"

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

set_seed(1)
rmb_label = {"1": 0, "100": 1}

MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1

# step 1/5: data preparation
split_dir = os.path.join(BASE_DIR, "..", "data", "rmb_split")
if not os.path.exists(split_dir):
    raise Exception(r"data {} not exist, go back to split_dataset.py to generate data".format(split_dir))
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 CenterCrop
    # transforms.CenterCrop(512),     # 512

    # 2 RandomCrop
    # transforms.RandomCrop(224, padding=16),
    # transforms.RandomCrop(224, padding=(16, 64)),
    # transforms.RandomCrop(224, padding=16, fill=(255, 0, 0)),
    # transforms.RandomCrop(512, pad_if_needed=True),   # pad_if_needed=True
    # transforms.RandomCrop(224, padding=64, padding_mode='edge'),
    # transforms.RandomCrop(224, padding=64, padding_mode='reflect'),
    # transforms.RandomCrop(1024, padding=1024, padding_mode='symmetric'),

    # 3 RandomResizedCrop
    # transforms.RandomResizedCrop(size=224, scale=(0.5, 0.5)),

    # 4 FiveCrop
    # transforms.FiveCrop(112),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 5 TenCrop
    # transforms.TenCrop(112, vertical_flip=False),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 1 Horizontal Flip
    # transforms.RandomHorizontalFlip(p=1),

    # 2 Vertical Flip
    # transforms.RandomVerticalFlip(p=0.5),

    # 3 RandomRotation
    # transforms.RandomRotation(90),
    # transforms.RandomRotation((90), expand=True),
    # transforms.RandomRotation(30, center=(0, 0)),
    # transforms.RandomRotation(30, center=(0, 0), expand=True),   # expand only for center rotation

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# build MyDataset instance
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# build DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# step 2/5: data preprocessing
for epoch in range(MAX_EPOCH):
    # Here the batch size is 1, so read the image one by one.
    for i, data in enumerate(train_loader):

        # Batch, Channel, Height, Width
        inputs, labels = data

        # Channel, Height, Width
        img_tensor = inputs[0, ...]
        img = transform_invert(img_tensor, train_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()


