# -*- coding: utf-8 -*-
"""
# @file name  : lesson-05-autograd.py
# @author     : Jianhua ma
# @date       : 2021-03-28
# @brief      : torch.autograd
"""
import torch
torch.manual_seed(10)

# torch.autograd.backward, obtain gradient automatically
# 1. retain_graph
# save computational graph
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # y.backward(retain_graph=True)
    # print(w.grad)
    y.backward(retain_graph=False)
    print(w.grad)
    print()


# 2. grad_tensors
# multiple gradient weights
# flag = True
flag = False

if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)     # retain_grad()
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)    # y0 = a * b, dy0/dw = 5
    y1 = torch.add(a, b)    # y1 = a + b, dy1/dw = 1 + 1 = 2

    loss = torch.cat([y0, y1], dim=0)   # [y0, y1]
    grad_tensors = torch.tensor([1., 2.])   # weight_y0 = 1, weight_y1 = 2

    loss.backward(gradient=grad_tensors)    # gradient is introduced to grad_tensors in the torch.autograd.backward()

    # w.grad:dy/dw . Here w.grad = 9 = 5*1 + 2*2
    print(w.grad)

# torch.autograd.grad, calculate the gradient
# 3. autograd.grad
# calculate the gradient
# flag = True
flag = False

if flag:
    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2) # y = x**2

    grad_1 = torch.autograd.grad(y, x, create_graph=True)   # grad_1 = dy/dx = 2x = 2*3 = 6
    print(grad_1)

    grad_2 = torch.autograd.grad(grad_1[0], x)  # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
    print(grad_2)

# tip 1
# gradient is not cleared automatically, so be careful about the clearing up
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        # if not grad.zero_()
        # i = 0, grad = 5
        # i = 1, grad = 5+5 =10
        y.backward()
        print(w.grad)

        w.grad.zero_()


# tip 2
# The nodes relied on leaf nodes set up requires_grad=True by default
flag = True
# flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)
    print(f"a grad:{a.grad}")

# tip 3
# The leaf node cannot implement in-place
# flag = True
flag = False
if flag:

    a = torch.ones((1, ))
    print(id(a), a)

    # return different address
    # a = a + torch.ones((1, ))
    # print(id(a), a)

    # return the same address
    a += torch.ones((1, ))
    print(id(a), a)

# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # An in-place operation is something which modifies the data of a variable. For example:
    # x += 1,  in-place
    # y = x + 1 , not in place
    # so .add_() and += cannot be implemented
    # w.add_(1)
    # w += 1
    w = w + 1

    y.backward()

"""
autograd tips:
1. 
2. 
3.  
"""
"""
Loosely, tensors you create directly are leaf variables. Tensors that are the result of a differentiable operation are 
not leaf variables

For example:

w = torch.tensor([1.0, 2.0, 3.0]) # leaf variable
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True) # also leaf variable
y = x + 1  # not a leaf variable
(The PyTorch documentation for is_leaf 156 contains a more precise definition.)

An in-place operation is something which modifies the data of a variable. For example:

x += 1  # in-place
y = x + 1 # not in place
PyTorch doesn’t allow in-place operations on leaf variables that have requires_grad=True (such as parameters of your 
model) because the developers could not decide how such an operation should behave. If you want the operation to 
be differentiable, you can work around the limitation by cloning the leaf variable (or use a non-inplace version 
of the operator).

x2 = x.clone()  # clone the variable
x2 += 1  # in-place operation
If you don’t intend for the operation to be differentiable, you can use torch.no_grad:

with torch.no_grad():
    x += 1
"""
