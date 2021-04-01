# -*- coding: utf-8 -*-
"""
# @file name  : nn_layers_others.py
# @author     : Jianhua Ma
# @date       : 20210331
# @brief      : other neural layers
"""

import os
import sys
import torch
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
# path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lena.png")
path_img = os.path.join(BASE_DIR, "..", "data", "lena.png")
img = Image.open(path_img).convert("RGB")   # 0~255

# convert to tensor
img_transform = transforms.Compose([
    transforms.ToTensor()
])
img_tensor = img_transform(img)
# add one dimension: C*H*W --> Batch*C*H*W, batch=1
img_tensor.unsqueeze_(dim=0)

# step 2: create other layers
# maxpool
# flag = True
flag = False
if flag:
    # input:(input channel, output channel, kernel size)
    # weights:(output channel, input channel, height, width)
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2))
    img_pool = maxpool_layer(img_tensor)

# avgpool
# flag = True
flag = False
if flag:
    avgpool_layer = nn.AvgPool2d((2, 2), stride=(2, 2))
    img_pool = avgpool_layer(img_tensor)

# avgpool divisor_override
# flag = True
flag = False
if flag:
    img_tensor = torch.ones((1, 1, 4, 4))
    # sum(elements in pooling map) / divisor_override
    # (1 + 1 + 1 + 1)/3 = 1.333
    avgpool_layer = nn.AvgPool2d((2, 2), stride=(2, 2), divisor_override=3)
    img_pool = avgpool_layer(img_tensor)

    print(f"raw_img:\n{img_tensor}\n, pooling_img:\n{img_pool}\n")

# max unpool
# flag = True
flag = False
if flag:
    # torch.randint: random integers with uniform distribution
    img_tensor = torch.randint(high=5, size=(1, 1, 4, 4), dtype=torch.float)
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
    img_pool, indices = maxpool_layer(img_tensor)

    # unpooling by the indices from the img_pool and img_tensor.
    # the img_pool cannot be reconstructed directly since the non-maximal values are lost.
    # maxunpooling can partially reconstruct the img_pool by the indices of maxial values in img_pool
    img_reconstruct = torch.randn_like(img_pool, dtype=torch.float)
    maxunpool_layer = nn.MaxUnpool2d((2, 2), stride=(2, 2))
    img_unpool = maxunpool_layer(img_reconstruct, indices)

    print(f"raw_img:\n{img_tensor}\nimg_pool:\n{img_pool}")
    print(f"img_reconstruct:\n{img_reconstruct}\nimg_unpool:\n{img_unpool}")

# Linear
flag = True
# flag = False
if flag:
    inputs = torch.tensor([
        [1., 2, 3]
    ])
    # input feature=3, output features = 4. Notice, the weight tensor should be (4, 3),not (3, 4).
    # if force the weight tensor to be (3, 4), the calculation will return error.
    linear_layer = nn.Linear(3, 4)
    linear_layer.weight.data = torch.tensor([
        [1., 1., 1.],
        [2., 2., 2.],
        [3., 3., 3.],
        [4., 4., 4.]
    ])

    # linear_layer.weight.data = torch.tensor([
    #     [1., 1., 1., 1.],
    #     [2., 2., 2., 2.],
    #     [3., 3., 3., 3.]
    # ])

    linear_layer.bias.data.fill_(0.5)
    output = linear_layer(inputs)
    print(inputs, inputs.shape)
    print(linear_layer.weight.data, linear_layer.weight.data.shape)
    print(output, output.shape)

# ================================= visualization ==================================
# print(f"img size before convolution is {img_tensor.shape}, after convolution is {img_pool.shape}")
# img_pool = transform_invert(img_pool[0, 0:3, ...], img_transform)
# img_raw = transform_invert(img_tensor.squeeze(), img_transform)
# plt.subplot(122).imshow(img_pool)
# plt.subplot(121).imshow(img_raw)
# plt.show()
