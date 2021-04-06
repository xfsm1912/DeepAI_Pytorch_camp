# -*- coding: utf-8 -*-
"""
# @file name  : module_containers.py
# @author     : Jianhua Ma
# @date       : 2021330
# @brief      : module container——Sequential, ModuleList, ModuleDict
"""
import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict


# steps of creating module
# 1. create module:
#   1.1 create network layer: __init__()
#   1.2 connect network layer: forward()
# 2. weight initialization

# nn.Module:
# 1. a nn.Module can contain multiple sub module
# 2. a module must implement forward() function
# 3. every module has 8 ordereddict to manage its characters

# self._parameters = OrderedDict(): store and manage nn.Parameter class
# self._buffers = OrderedDict(): store and manage buffer characters
# self._backward_hooks = OrderedDict()
# self._forward_hooks = OrderedDict()
# self._forward_pre_hooks = OrderedDict()
# self._state_dict_hooks = OrderedDict()
# self._load_state_dict_pre_hooks = OrderedDict()
# self._modules = OrderedDict(): store and manage nn.Module class

# containers 1: nn.Sequential()
class LeNetSequential(nn.Module):
    def __init__(self, classes):
        super(LeNetSequential, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)

        return x


class LeNetSequentialOrderDict(nn.Module):
    def __init__(self, classes):
        super(LeNetSequentialOrderDict, self).__init__()
        # input is a OrderedDict({})
        self.features = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 6, 5),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),

            'conv2': nn.Conv2d(6, 16, 5),
            'relu2': nn.ReLU(inplace=True),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
        }))

        self.classifier = nn.Sequential(OrderedDict({
            'fc1': nn.Linear(16 * 5 * 5, 120),
            'relu3': nn.ReLU(),

            'fc2': nn.Linear(120, 84),
            'relu4': nn.ReLU(inplace=True),

            'fc3': nn.Linear(84, classes),
        }))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


# train = True
train = False
if train:
    net = LeNetSequential(classes=2)
    # netdict = LeNetSequentialOrderDict(classes=2)

    fake_img = torch.randn((4, 3, 32, 32), dtype=torch.float32)

    output = net(fake_img)
    # output = netdict(fake_img)

    print(net)
    print(output)


# container 2: ModuleList
class ModuleList(nn.Module):
    def __init__(self):
        super(ModuleList, self).__init__()
        self.linear = nn.ModuleList([
            nn.Linear(10, 10) for _ in range(20)
        ])

    def forward(self, x):
        for _, linear in enumerate(self.lienar):
            x = linear(x)
        return x


# train = True
train = False
if train:
    net = ModuleList()
    print(net)

    fake_data = torch.ones((10, 10))
    output = net(fake_data)
    print(output)


# container 4: ModuleDict
class ModuleDict(nn.Module):
    def __init__(self):
        super(ModuleDict, self).__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'prelu': nn.PReLU()
        })

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x


train = True
# train = False
if train:
    net = ModuleDict()
    fake_img = torch.randn((4, 10, 32, 32))
    output = net(fake_img, 'conv', 'relu')
    print(output.shape)

# 4 AlexNet
# alexnet = torchvision.models.AlexNet()

