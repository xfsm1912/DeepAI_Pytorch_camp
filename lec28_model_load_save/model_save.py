# -*- coding: utf-8 -*-
"""
# @file name  : model_save.py
# @author     : Jianhua
# @date       : 20210403
# @brief      : save a model
"""
import torch
import numpy as np
import torch.nn as nn

from model.lenet import LeNet2

net = LeNet2(classes=2021)

# training
print(f"before training: {net.features[0].weight[0, ...]}")
net.initialize()
print(f"after training: {net.features[0].weight[0, ...]}")

path_model = "./model.pkl"
path_state_dict = "./model_state_dict.pkl"

# save model
torch.save(net, path_model)
# save model parameter
net_state_dict = net.state_dict()
torch.save(net_state_dict, path_state_dict)

