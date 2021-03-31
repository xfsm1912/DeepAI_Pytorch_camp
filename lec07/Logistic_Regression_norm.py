# -*- coding: utf-8 -*-
"""
# @file name  : Logsitic-Regression-norm.py
# @author     : Jianhua Ma
# @date       : 20210329
# @brief      :
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)

lr = 0.01  # learning rate

# data generation
sample_nums = 100
mean_value = 1.7
bias = 5         # 5
n_data = torch.ones(sample_nums, 2)
# class 0, data: shape=(100, 2)
# class 0, label: shape=(100, 1)
x0 = torch.normal(mean_value * n_data, 1) + bias
# class 0, label: shape=(100)
# y0 = torch.zeros(sample_nums)
y0 = torch.zeros((sample_nums, 1))
x1 = torch.normal(-mean_value * n_data, 1) + bias
# y1 = torch.ones(sample_nums)
y1 = torch.ones((sample_nums, 1))
train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)


# model
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x


lr_net = LR()

# loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(lr_net.parameters(), lr=0.01, momentum=0.9)

for iteration in range(1000):

    # forward propagation
    y_pred = lr_net(train_x)

    # MSE loss
    # if y0 = torch.zeros(sample_nums)
    # loss = loss_fn(y_pred, train_y.unsqueeze(1))
    # if y0 = torch.zeros((sample_nums, 1))
    loss = loss_fn(y_pred, train_y)
    """
    if not use unsqueeze(1)，will return error as below:
    Using a target size (torch.Size([200])) that is different to the input size (torch.Size([200, 1])) is deprecated. 
    Please ensure they have the same size.
    """

    # back propagation
    loss.backward()

    # update weights
    optimizer.step()

    # clear gradient
    optimizer.zero_grad()

    # plot learning curve
    if iteration % 40 == 0:
        plt.clf()

        # if y0 = torch.zeros(sample_nums)
        # mask = y_pred.ge(0.5).float().squeeze()
        # if y0 = torch.zeros((sample_nums, 1))
        mask = y_pred.ge(0.5).float()
        correct = (mask == train_y).sum()
        acc = correct.item() / train_y.size()[0]

        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 10)
        plt.ylim(-7, 10)
        plt.plot(plot_x, plot_y)

        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
        plt.legend()

        plt.show()
        plt.pause(0.5)

        if acc > 0.99:
            break
