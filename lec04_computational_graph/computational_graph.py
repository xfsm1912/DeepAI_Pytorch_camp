# -*- coding:utf-8 -*-
"""
@file name  : lesson-04-Computational-Graph.py
@author     : Jianhua Ma
@date       : 2021-03-28
@brief      : computational graph example
"""

import torch
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

# y = (x + w) * (w + 1) = a * b
a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward()
print(w.grad)

# check graph leaves
# print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# check gradient
# w.grad: dy/dw,
# x.grad: dy/dx.
# print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)

# check grad_fn
print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)
