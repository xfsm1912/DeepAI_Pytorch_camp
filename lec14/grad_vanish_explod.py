# -*- coding: utf-8 -*-
"""
# @file name  : grad_vanish_explod.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-09-30 10:08:00
# @brief      : 梯度消失与爆炸实验
"""
import os

import torch
import sys
import random
import numpy as np
import torch.nn as nn
from tools.common_tools import set_seed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_tools = os.path.abspath(os.path.join(BASE_DIR, "..", "tools", "common_tools.py"))
assert os.path.exists(
    path_tools), f"{path_tools} not exist, please place common_tools.py in the {os.path.dirname(path_tools)}"

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

set_seed(1)


class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        super(MLP, self).__init__()
        self.linear = nn.ModuleList([
            nn.Linear(neural_num, neural_num, bias=False) for _ in range(layers)
        ])

        self.neural_num = neural_num

    def forward(self, x):
        for (i, linear) in enumerate(self.linear):
            x = linear(x)
            # x = torch.tanh(x)
            x = torch.relu(x)

            print(f"layer: {i}, std:{x.std()}")
            if torch.isnan(x.std()):
                print(f"output is nan in {i} layers")
                break

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                ###
                # task 1: x = linear(x), gradient explosion
                # compare normal initialization and std = 1/sqrt(n)

                # nn.init.normal_(m.weight.data)  # std increase from 16, 256, ...
                # nn.init.normal_(m.weight.data, std=np.sqrt(1 / self.neural_num))  # normal: mean=0, std=1, std keep in 1

                ###
                # task 2: x = linear(x), x = torch.tanh(x)
                # if initialization is normal_, will gradient vanish
                # so W in [sqrt(6)/sqrt(n+n)]

                # 1. manual calculate the a
                # a = np.sqrt(6 / (self.neural_num + self.neural_num))
                # tanh_gain = nn.init.calculate_gain('tanh')
                # a *= tanh_gain
                #
                # nn.init.uniform_(m.weight.data, -a, a)  # std keep in 0.65

                # 2. official version
                # nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)  # std keep in 0.65

                # task 3: x = linear(x), x = torch.relu(x)
                # nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.neural_num))
                nn.init.kaiming_normal_(m.weight.data)


flag = False
# flag = True

if flag:
    layer_nums = 100
    neural_nums = 256
    batch_size = 16

    net = MLP(neural_nums, layer_nums)
    net.initialize()

    inputs = torch.randn((batch_size, neural_nums))  # normal: mean=0, std=1

    output = net(inputs)
    print(output)

# ======================================= calculate gain =======================================
# input std / output std
# flag = False
flag = True

if flag:
    x = torch.randn(10000)
    out = torch.tanh(x)

    gain = x.std() / out.std()
    print('gain:{}'.format(gain))

    tanh_gain = nn.init.calculate_gain('tanh')
    print('tanh_gain in PyTorch:', tanh_gain)
