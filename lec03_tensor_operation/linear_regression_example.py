# -*- coding:utf-8 -*-
"""
@file name  : lesson-03-Linear-Regression.py
@author     : Jianhua Ma
@date       : 2021-03-28
@brief      : one variable linear regression
"""

import torch
import matplotlib.pyplot as plt
torch.manual_seed(10)

lr = 0.05

# uniform distribution
x = torch.rand(20, 1) * 10
# normal distribution
y = 2 * x + (5 + torch.randn(20, 1))

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

for iteration in range(1000):

    # forward propagation
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # calculate mse los
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # backward propagation
    # backward: Computes the gradient of current tensor w.r.t. graph leaves.
    loss.backward()

    # update weights
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # clear tensor gradient
    w.grad.zero_()
    b.grad.zero_()

    # plot
    if iteration % 20 == 0:

        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)

        if loss.data.numpy() < 1:
            break
