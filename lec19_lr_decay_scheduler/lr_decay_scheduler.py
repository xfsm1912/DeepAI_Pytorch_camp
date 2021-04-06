# -*- coding:utf-8 -*-
"""
@file name  : lr_decay_scheduler.py
@author     : Jianhua Ma
@date       : 20210331
@brief      : learning rate decay scheduler
"""
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

LR = 0.1
iteration = 10
max_epoch = 200
# ------------------------------ fake data and optimizer  ------------------------------

weights = torch.randn(1, requires_grad=True)
target = torch.zeros(1)

# the updated parameter is weight
optimizer = optim.SGD([weights], lr=LR, momentum=0.9)

# example 1: Step LR, StepLR()
flag = 0
# flag = 1
if flag:

    # lr = lr * gamma after each 50 iteration steps period
    scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  #

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):

        # obtain current lr，new version is get_last_lr()，old version is get_lr()，based on your UserWarning
        lr_list.append(scheduler_lr.get_last_lr())
        epoch_list.append(epoch)

        for i in range(iteration):
            # (w - t)^2 --> w = w - gamma * 2 * (w - t) = w - 0.1 * 2 w = 0.8*w
            loss = torch.pow((weights - target), 2)
            # gradient
            loss.backward()

            # update the weight
            optimizer.step()
            optimizer.zero_grad()

        # update the learning rate after one epoch
        scheduler_lr.step()

    plt.plot(epoch_list, lr_list, label="Step LR Scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()

# example 2: Multi Step LR
flag = 0
# flag = 1
if flag:

    # learning rate will change in step 50, 125, and 160
    milestones = [50, 125, 160]
    scheduler_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):

        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):
            loss = torch.pow((weights - target), 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list, label="Multi Step LR Scheduler\nmilestones:{}".format(milestones))
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()

# example 3: Exponential LR
flag = 0
# flag = 1
if flag:

    # learning rate decrease as steps exponentially
    # lr = lr * gamma^(epoch)
    gamma = 0.95
    scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):

        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):
            loss = torch.pow((weights - target), 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list, label="Exponential LR Scheduler\ngamma:{}".format(gamma))
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()

# example 4: Cosine Annealing LR
flag = 0
# flag = 1
if flag:
    # learning rate change with cosine period
    t_max = 50
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.)

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):

        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):
            loss = torch.pow((weights - target), 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list, label="CosineAnnealingLR Scheduler\nT_max:{}".format(t_max))
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()

# example 5: Reduce LR On Plateau
# monitor the learning rate, if learning rate doesn't change, then modify
flag = 0
# flag = 1
if flag:
    loss_value = 0.5
    accuray = 0.9

    factor = 0.1
    mode = "min"
    patience = 10
    cooldown = 10
    min_lr = 1e-4
    verbose = True

    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, mode=mode, patience=patience,
                                                        cooldown=cooldown, min_lr=min_lr, verbose=verbose)

    for epoch in range(max_epoch):
        for i in range(iteration):
            # train(...)

            optimizer.step()
            optimizer.zero_grad()

        if epoch == 5:
            loss_value = 0.4

        scheduler_lr.step(loss_value)

# example 6: lambda
# define the learning rate by yourself
flag = 0
# flag = 1
if flag:

    lr_init = 0.1

    weights_1 = torch.randn((6, 3, 5, 5))
    weights_2 = torch.ones((5, 5))

    optimizer = optim.SGD([
        {'params': [weights_1]},
        {'params': [weights_2]}], lr=lr_init)

    lambda1 = lambda epoch: 0.1 ** (epoch // 20)
    lambda2 = lambda epoch: 0.95 ** epoch

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    lr_list, epoch_list = list(), list()
    for epoch in range(max_epoch):
        for i in range(iteration):
            # train(...)

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        lr_list.append(scheduler.get_lr())
        epoch_list.append(epoch)

        print('epoch:{:5d}, lr:{}'.format(epoch, scheduler.get_lr()))

    plt.plot(epoch_list, [i[0] for i in lr_list], label="lambda 1")
    plt.plot(epoch_list, [i[1] for i in lr_list], label="lambda 2")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("LambdaLR")
    plt.legend()
    plt.show()
