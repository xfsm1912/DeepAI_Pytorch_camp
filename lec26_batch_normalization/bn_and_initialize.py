# -*- coding: utf-8 -*-
"""
# @file name  : bn_and_initialize.py
# @author     : Jianhua Ma
# @date       : 20210403
# @brief      : batch normalization and weight initialization
"""
import torch
import numpy as np
import torch.nn as nn
from tools.common_tools import set_seed
import sys
import os
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

set_seed(1)


# By setting up a lot of hidden layers, we can see how the weights change as layer increasing.
class MLP(nn.Module):
    def __init__(self, neural_num, layers=100):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([
            nn.Linear(neural_num, neural_num, bias=False) for _ in range(layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(neural_num) for _ in range(layers)]
        )
        self.neural_num = neural_num

    def forward(self, x):

        for (i, linear), bn in zip(enumerate(self.linears), self.bns):
            x = linear(x)
            x = bn(x)
            x = torch.relu(x)

            if torch.isnan(x.std()):
                print(f"output is nan in {i} layers")
                break

            print(f"layers:{i}, std:{x.std().item()}")

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):

                # method 1
                # nn.init.normal_(m.weight.data, std=1)    # normal: mean=0, std=1

                # method 2 kaiming
                nn.init.kaiming_normal_(m.weight.data)


neural_nums = 256
layer_nums = 100
batch_size = 16

net = MLP(neural_nums, layer_nums)
# net.initialize()

inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1

output = net(inputs)
print(output)
