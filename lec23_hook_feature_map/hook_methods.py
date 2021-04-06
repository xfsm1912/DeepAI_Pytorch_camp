# -*- coding:utf-8 -*-
"""
@file name  : hook_methods.py
@author     : Jianhua Ma
@date       : 20210402
@brief      : hook function
"""

import torch
import torch.nn as nn

import sys, os
from tools.common_tools import set_seed

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

set_seed(1)

# example 1 : tesnor hook
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # a_grad = []
    b_grad = []

    def grad_hook(grad):
        # a_grad.append(grad)
        b_grad.append(grad)

    # handle = a.register_hook(grad_hook)
    handle = b.register_hook(grad_hook)

    y.backward()

    # check gradient
    print(f"gradient:{w.grad}, {x.grad}, {a.grad}, {b.grad}, {y.grad}")
    # dy/da = b = w + 1 = 2
    # print(f"a_grad[0]: {a_grad[0]}")
    # dy/db = a = x + w = 3
    print(f"b_grad[0]: {b_grad[0]}")
    handle.remove()


# example 2: tensor hook 2
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    a_grad = []

    def grad_hook(grad):
        grad *= 2
        return grad*3

    hanlde = w.register_hook(grad_hook)

    y.backward()

    # check gradient
    # dy/dw = 5 * 2 * 3
    print(f"w.grad: {w.grad}")
    hanlde.remove()

# example 3: Module.register_forward_hook and pre hook
flag = True
# flag = False
if flag:

    feature_map_block, input_block = [], []

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 2, 3)
            self.pool1 = nn.MaxPool2d(2, 2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            return x

    def foward_hook(module, data_input, data_output):
        feature_map_block.append(data_output)
        input_block.append(data_input)

    def forward_pre_hook(module, data_input):
        print(f"forward_pre_hook input:{data_input}")

    def backward_hook(module, grad_input, grad_output):
        print(f"backward hook input:{grad_input}")
        print(f"backward hook output:{grad_output}")

    # initialize the network
    net = Net()
    net.conv1.weight[0].detach().fill_(1)
    net.conv1.weight[1].detach().fill_(2)
    net.conv1.bias.data.detach().zero_()

    # register hook
    net.conv1.register_forward_hook(foward_hook)
    net.conv1.register_forward_pre_hook(forward_pre_hook)
    net.conv1.register_backward_hook(backward_hook)

    # inference
    # batch size: batch * channel * height * width
    fake_img = torch.ones((1, 1, 4, 4))
    output = net(fake_img)

    loss_fnc = nn.L1Loss()
    target = torch.rand_like(output)
    loss = loss_fnc(target, output)
    loss.backward()

    print(f"output shape: {output.shape}\noutput value: {output}\n")
    print(f"feature maps shape: {feature_map_block[0].shape}\noutput value: {feature_map_block[0]}\n")
    print(f"input shape: {input_block[0][0].shape}\ninput value: {input_block[0]}")


