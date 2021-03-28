# -*- coding:utf-8 -*-
"""
@file name  : lesson-02.py
@author     : Jianhua Ma
@date       : 2021-03-28
@brief      : create tensor
"""

import torch
import numpy as np

# Sets the seed for generating random numbers. Returns a torch.Generator
torch.manual_seed(1)

cuda_available = torch.cuda.is_available()

# Example 1
# create tensor by torch.tensor

flag = False
# flag = True
if flag:
    arr = np.ones((3, 3))
    print("The data type of ndarray: arr.dtype")

    if cuda_available:
        t = torch.tensor(arr, device='cuda')
    else:
        t = torch.tensor(arr)

    print(t)

# Example 2
# create by torch.from_numpy

# flag = True
flag = False
if flag:
    arr = np.array(
        [[1, 2, 3],
         [4, 5, 6]]
    )
    t = torch.from_numpy(arr)
    print(f"numpy array: \n{arr}")
    print(f"tensor: \n{t}")

    # the ndarray and tensor are linked with each other, changes in the elements in each side will affect the other
    # side.
    print("\nchange arr")
    arr[0, 0] = 0
    print(f"numpy array: \n{arr}")
    print(f"tensor: \n{t}")

    print("\nchange tensor")
    t[0, 0] = -1
    print(f"numpy array: \n{arr}")
    print(f"tensor: \n{t}")

# Example 3
# create by torch.zero

# flag = True
flag = False
if flag:
    out_t = torch.tensor([1])

    # param: out=out_t, the output variable becomes out_t, so it becomes a 3*3 tensor instead of tensor([1])
    t = torch.zeros((3, 3), out=out_t)
    # torch.zeros((3, 3), out=out_t)
    # t = torch.zeros((3, 3))

    print(t, '\n', out_t)
    print(id(t), id(out_t), id(t) == id(out_t))

# Example 4
# all 1 tensor created by torch.full
flag = False
# flag = True

if flag:
    t = torch.full((3, 3), 1)
    t2 = torch.full((3, 3), 2)
    print(t)
    print(t2)

# Exampel 5
# create Arithmetic progression by torch.arrange
flag = False
# flag = True

if flag:
    t = torch.arange(2, 10, 2)
    print(t)

# Example 6
# create series tensor by torch.linspace
# flag = True
flag = False
if flag:
    t = torch.linspace(2, 10, 5)
    t2 = torch.linspace(2, 10, 6)
    print(t)
    print(t2)

# Example 7
# normal distribution tensor created by torch.normal
flag = True
# flag = False

if flag:
    # mean：tensor std: tonsor
    # mean = torch.arange(1, 5, dtype=torch.float)
    # std = torch.arange(1, 5, dtype=torch.float)
    # t_normal = torch.normal(mean, std)
    # print("mean:{}\nstd:{}".format(mean, std))
    # print(t_normal)

    # mean：scalar std: scalar
    t_normal = torch.normal(0., 1., size=(4,))
    print(t_normal)

    # mean：tensor std: scalar
    # mean = torch.arange(1, 5, dtype=torch.float)
    # std = 1
    # t_normal = torch.normal(mean, std)
    # print("mean:{}\nstd:{}".format(mean, std))
    # print(t_normal)

