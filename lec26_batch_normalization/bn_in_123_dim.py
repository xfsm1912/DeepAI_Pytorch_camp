# -*- coding: utf-8 -*-
"""
# @file name  : bn_in_123_dim.py
# @author     : Jianhua Ma
# @date       : 20210403
# @brief      : three kinds of bn functions for different dimension data.
"""
import torch
import numpy as np
import torch.nn as nn
import sys
import os
from tools.common_tools import set_seed

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)


set_seed(1)

# example 1: nn.BatchNorm1d
# flag = True
flag = False
if flag:
    batch_size = 3
    num_features = 5
    momentum = 0.3

    features_shape = 1

    # 1d
    feature_map = torch.ones(features_shape)
    # 1d --> 2d
    # [1], [2], [3], [4], [5]
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)
    # 2d --> 3d
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)
    print(f"input data:\n {feature_maps_bs} shape is {feature_maps_bs.shape}")

    bn = nn.BatchNorm1d(num_features=num_features, momentum=momentum)

    running_mean, running_var = 0, 1

    for i in range(2):
        outputs = bn(feature_maps_bs)

        print(f"\niterations: {i}, running mean: {bn.running_mean}")

        # check the second feature, the second feature is initialized as 2
        mean_t, var_t = 2, 0
        running_mean = (1 - momentum) * running_mean + momentum * mean_t
        running_var = (1 - momentum) * running_var + momentum * var_t

        print(f"iteration:{i}, running mean of the second feature: {running_mean} ")
        print(f"iteration:{i}, running var of the second feature:{running_var}")

# example 2: nn.BatchNorm2d
# flag = True
flag = False
if flag:

    batch_size = 3
    num_features = 6
    momentum = 0.3

    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)  # 2D
    feature_maps = torch.stack([feature_map * (i + 1) for i in range(num_features)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for i in range(batch_size)], dim=0)  # 4D

    print("input data:\n{} shape is {}".format(feature_maps_bs, feature_maps_bs.shape))

    bn = nn.BatchNorm2d(num_features=num_features, momentum=momentum)

    running_mean, running_var = 0, 1

    for i in range(2):
        outputs = bn(feature_maps_bs)

        print("\niter:{}, running_mean.shape: {}".format(i, bn.running_mean.shape))
        print("iter:{}, running_var.shape: {}".format(i, bn.running_var.shape))

        print("iter:{}, weight.shape: {}".format(i, bn.weight.shape))
        print("iter:{}, bias.shape: {}".format(i, bn.bias.shape))

# example 3: nn.BatchNorm3d
flag = True
# flag = False
if flag:

    batch_size = 3
    num_features = 4
    momentum = 0.3

    features_shape = (2, 2, 3)

    feature = torch.ones(features_shape)  # 3D
    feature_map = torch.stack([feature * (i + 1) for i in range(num_features)], dim=0)  # 4D
    feature_maps = torch.stack([feature_map for i in range(batch_size)], dim=0)  # 5D

    print("input data:\n{} shape is {}".format(feature_maps, feature_maps.shape))

    bn = nn.BatchNorm3d(num_features=num_features, momentum=momentum)

    running_mean, running_var = 0, 1

    for i in range(2):
        outputs = bn(feature_maps)

        print("\niter:{}, running_mean.shape: {}".format(i, bn.running_mean.shape))
        print("iter:{}, running_var.shape: {}".format(i, bn.running_var.shape))

        print("iter:{}, weight.shape: {}".format(i, bn.weight.shape))
        print("iter:{}, bias.shape: {}".format(i, bn.bias.shape))

