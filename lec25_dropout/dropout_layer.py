# -*- coding:utf-8 -*-
"""
@file name  : dropout_layer.py
# @author   : Jianhua Ma
@date       : 20210403
@brief      : dropout test for training mode and evaluation mode
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys
from tools.common_tools import set_seed

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

set_seed(1)


class Net(nn.Module):
    def __init__(self, neural_num, d_prob=0.5):
        super(Net, self).__init__()
        self.linears = nn.Sequential(
            nn.Dropout(d_prob),
            nn.Linear(neural_num, 1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.linears(x)


input_num = 10000
x = torch.ones((input_num, ), dtype=torch.float32)

# here the input nodes number is 10000, the hidden node weight is 1
# so the output should be sum(1*1, 10000) = 10000
net = Net(input_num, d_prob=0.5)
net.linears[1].weight.detach().fill_(1.)

# Sets the module in training mode.
# In pytorch, we do the weight scaling in training stage by w / (1-p_drop) = w/(1-0.5) = 2w = 2
# and 50% of nodes are dropout in training, the output is sum(2*1, 5000) = 10000
# in evaluation mode, the prediction doesn't need to do the weight scaling, the output is
# sum(w*x, 10000) = sum(1*1, 10000) = 10000.
net.train()
y = net(x)
print(f"output in the training mode is {y}")

net.eval()
y = net(x)
print(f"output in the evaluation mode is {y}")



