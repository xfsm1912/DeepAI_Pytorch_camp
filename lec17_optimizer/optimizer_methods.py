# -*- coding: utf-8 -*-
"""
# @file name  : optimizer_methods.py
# @author     : Jianhua Ma
# @date       : 20210331
# @brief      : optimizer's methods
"""
import os

import torch
import torch.optim as optim

import sys
from tools.common_tools import set_seed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

set_seed(1)  # 设置随机种子

weight = torch.randn((2, 2), requires_grad=True)
weight.grad = torch.ones((2, 2))

optimizer = optim.SGD([weight], lr=0.1)

# example 1: step()
flag = 0
# flag = 1
if flag:
    print(f"weight before step:{weight.data}")
    optimizer.step()        # we can change lr=1, 0.1, the weight.data will change by 1, 0.1
    print(f"weight after step:{weight.data}")


# example 2: zero_grad(), pytorch does not clear up its gradient by default.
flag = 0
# flag = 1
if flag:
    print("weight before step:{}".format(weight.data))
    optimizer.step()        # we can change lr=1, 0.1, the weight.data will change by 1, 0.1
    print("weight after step:{}".format(weight.data))

    print("weight id in optimizer:{}\nweight id in weight:{}\n".format(id(optimizer.param_groups[0]['params'][0]),
                                                                       id(weight)))

    print("weight.grad is {}\n".format(weight.grad))
    optimizer.zero_grad()
    print("after optimizer.zero_grad(), weight.grad is\n{}".format(weight.grad))

# example 3: add_param_group
flag = 0
# flag = 1
if flag:
    print(f"optimizer.param_groups is\n{optimizer.param_groups}")

    w2 = torch.randn((3, 3), requires_grad=True)

    optimizer.add_param_group({"params": w2, 'lr': 0.0001})

    print(f"optimizer.param_groups is\n{optimizer.param_groups}")

# example 4: state_dict, get the current info dict in optimizer.
flag = 0
# flag = 1
if flag:

    optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
    opt_state_dict = optimizer.state_dict()

    print(f"state_dict before step:{opt_state_dict}\n")

    for i in range(10):
        optimizer.step()

    print(f"state_dict after step:{optimizer.state_dict()}\n")

    torch.save(optimizer.state_dict(), os.path.join(BASE_DIR, "optimizer_state_dict.pkl"))

# example 5: load_state_dict
# flag = 0
flag = 1
if flag:

    optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
    state_dict = torch.load(os.path.join(BASE_DIR, "optimizer_state_dict.pkl"))

    print("state_dict before load state:\n", optimizer.state_dict())
    optimizer.load_state_dict(state_dict)
    print("state_dict after load state:\n", optimizer.state_dict())
