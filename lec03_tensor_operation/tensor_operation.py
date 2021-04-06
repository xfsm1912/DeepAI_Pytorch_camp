# -*- coding:utf-8 -*-
"""
@file name  : lesson-03.py
@author     : Jianhua Ma
@date       : 2021-03-28
@brief      : tensor operation
"""

import torch
torch.manual_seed(1)

# Example 1
# torch.cat

# flag = True
flag = False
if flag:
    t = torch.ones((2, 3))

    t_0 = torch.cat([t, t], dim=0)
    t_1 = torch.cat([t, t, t], dim=1)

    print(f"t_0:{t_0} shape: {t_0.shape}\n t_1:{t_1} shape:{t_1.shape}")

# Example 2
# torch.stack
# flag = True
flag = False

if flag:
    t = torch.ones((2, 3))
    t_stack = torch.stack([t, t, t], dim=0)

    print(f"\nt_stack:{t_stack} shape:{t_stack.shape}")

# Example 3
# split a tensor into a specific number of chunks by torch.chunk

# flag = True
flag = False
if flag:
    a = torch.ones((2, 7))
    list_of_tensors = torch.chunk(a, dim=1, chunks=3)

    for idx, t in enumerate(list_of_tensors):
        print(f"No.{idx} tensor is {t}, shape is {t.shape}")

# Example 4
# split a tensor into chunk by torch.split
# flag = True
flag = False
if flag:
    t = torch.ones((2, 5))

    # the split_size_or_sections is a list including the sizes for each chunk
    list_of_tensors = torch.split(t, split_size_or_sections=[2, 1, 2], dim=1)
    # list_of_tensors = torch.split(t, [2, 1, 1], dim=1)
    for idx, t in enumerate(list_of_tensors):
        print(f"No.{idx} tensor is {t}, shape is {t.shape}")

# Example 5
# select part of the tensor by torch.index_select
# flag = True
flag = False

if flag:
    t = torch.randint(0, 9, size=(3, 3))
    idx = torch.tensor([0, 2], dtype=torch.long)
    # index_select(): Expected dtype int32 or int64 for index
    # idx = torch.tensor([0, 2], dtype=torch.float32)
    t_select = torch.index_select(t, dim=0, index=idx)
    print(f"t:\n{t}\nt_select from index {idx}:\n{t_select}")

# Example 6
# torch.masked_select

# flag = True
flag = False

if flag:

    t = torch.randint(0, 9, size=(3, 3))
    mask = t.le(5)  # ge is mean greater than or equal/   gt: greater than  le  lt
    t_select = torch.masked_select(t, mask)
    print("t:\n{}\nmask:\n{}\nt_select:\n{} ".format(t, mask, t_select))

# Example 7
# torch.reshape

# flag = True
flag = False
if flag:
    t = torch.randperm(8)
    t_reshape = torch.reshape(t, (-1, 2, 2))
    print(f"t:{t}\nt_reshape:{t_reshape}\n")

    t[0] = 1024
    print(f"t:{t}\nt_reshape:{t_reshape}\n")
    # the t and t_reshape share the same address
    print(f"t.data memory address:{id(t.data)}")
    print(f"t_reshape.data memory address:{id(t_reshape.data)}")

# Example 8
# torch.transpose
flag = False
# flag = True

if flag:
    # torch.transpose
    t = torch.rand((2, 3, 4))
    # c*h*w --> c*w*h. Here we tranpose the second and third dimension
    t_transpose = torch.transpose(t, dim0=1, dim1=2)
    print(f"t shape:{t.shape}\nt_transpose shape:{t_transpose.shape}")

# Example 9
# remove the dimensions of tensor of size 1 by torch.squeeze
# flag = True
flag = False

if flag:
    t = torch.rand((1, 2, 3, 4))
    t_sq = torch.squeeze(t)
    # since size in dim=0 is 1, the size along dim=0 removed
    t_0 = torch.squeeze(t, dim=0)
    # since size in dim=1 is 2, nothing change
    t_1 = torch.squeeze(t, dim=1)
    print(t.shape)
    print(t_sq.shape)
    print(t_0.shape)
    print(t_1.shape)

# Example 9
# torch.add
flag = True
# flag = True

if flag:
    t_0 = torch.randn((3, 3))
    # return a tensor filled with the scalar value 1,  same size with input
    t_1 = torch.ones_like(t_0)
    # output = input + alpha * other
    t_add = torch.add(t_0, t_1, alpha=10)
    print("t_0:\n{}\nt_1:\n{}\nt_add_10:\n{}".format(t_0, t_1, t_add))
