# -*- coding: utf-8 -*-
"""
# @file name  : model_load.py
# @author     : Jianhua Ma
# @date       : 20210403
# @brief      : model loading and saving
"""
import torch
import numpy as np
import torch.nn as nn

from model.lenet import LeNet2

# Before run this script, run the model_save.py first
# example 1: load net
# flag = True
flag = False
if flag:
    path_model = "./model.pkl"
    net_load = torch.load(path_model)

    print(net_load)

# example 2: load state_dict

# flag = True
flag = False
if flag:

    path_state_dict = "./model_state_dict.pkl"
    state_dict_load = torch.load(path_state_dict)

    print(state_dict_load.keys())

# example 3: update state_dict
# flag = True
flag = False
if flag:

    net_new = LeNet2(classes=2021)

    print("before loading: ", net_new.features[0].weight[0, ...])
    net_new.load_state_dict(state_dict_load)
    print("after loading: ", net_new.features[0].weight[0, ...])
