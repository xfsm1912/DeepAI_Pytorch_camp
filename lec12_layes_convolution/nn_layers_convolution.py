# -*- coding: utf-8 -*-
"""
# @file name  : nn_layers_convolution.py
# @author     : Jianhua Ma
# @date       : 20210331
# @brief      : convolution layer
"""

import os
import sys

import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from tools.common_tools import transform_invert, set_seed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_tools = os.path.abspath(os.path.join(BASE_DIR, "..", "tools", "common_tools.py"))
assert os.path.exists(path_tools), f"{path_tools} not exist, please place common_tools.py in the {os.path.dirname(path_tools)}"
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

set_seed(3)

# step 1: load image
path_img = os.path.join(BASE_DIR, "..", "data", "lena.png")
img = Image.open(path_img).convert("RGB")   # 0~255

# convert to tensor
img_transform = transforms.Compose([
    transforms.ToTensor()
])
img_tensor = img_transform(img)
# add one dimension: C*H*W --> Batch*C*H*W, batch=1
img_tensor.unsqueeze_(dim=0)

# step 2: create
# 2d convolution layer
# flag = True
flag = False
if flag:
    # input:(input channel, output channel, kernel size) = (3, 1, 3)
    # weights:(output channel, input channel, height, width) = (1, 3, 3, 3)
    conv_layer = nn.Conv2d(3, 1, 3)
    nn.init.xavier_normal_(conv_layer.weight.data)

    img_conv = conv_layer(img_tensor)

# transposed convolution: upsampling the image
# https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers
flag = True
# flag = False
if flag:
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride=2)
    nn.init.xavier_normal_(conv_layer.weight.data)

    img_conv = conv_layer(img_tensor)

print(f"img size before convolution is {img_tensor.shape}, after convolution is {img_conv.shape}")
img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)   # only present one channel
img_raw = transform_invert(img_tensor.squeeze(), img_transform)     # remove the batch dimension by squeeze()
plt.subplot(122).imshow(img_conv, cmap='gray')
plt.subplot(121).imshow(img_raw)
plt.show()


