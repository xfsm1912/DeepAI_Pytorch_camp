# -*- coding: utf-8 -*-
"""
# @file name  : create_module.py
# @author     : Jianhua Ma
# @date       : 20210330
# @brief      : create new transform class
"""

import os
import numpy as np
import torch
import random
from PIL import Image
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


class AddPepperNoise(object):
    """
    add pepper noise into image
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """

        :param img: PIL image
        :return: PIL image
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            # random generate a matrix with size of (h, w, 1), the value is 0, 1, or 2.The probability
            # is 0.9, 0.05 and 0.05
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # salty noise
            img_[mask == 2] = 0     # pepper noise
            return Image.fromarray(img_.astype("uint8")).convert("RGB")
        else:
            return img


# step 1/5: data preparation
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
    AddPepperNoise(0.9, p=0.5),
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
