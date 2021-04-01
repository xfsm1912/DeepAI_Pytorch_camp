# -*- coding: utf-8 -*-
"""
# @file name  : RMB_data_augmentation.py
# @author     : Jianhua Ma
# @date       : 20210330
# @brief      : Based on lec06/train_lenet.py, add the data augmentation.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
import sys
from model.lenet import LeNet
from tools.my_dataset import RMBDataset
from tools.common_tools import set_seed, transform_invert

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

path_lenet = os.path.join(BASE_DIR, "..", "model", "lenet.py")
path_tools = os.path.join(BASE_DIR, "..", "tools", "common_tools.py")

assert os.path.exists(path_lenet), f"{path_lenet} not exist, please place lenet.py in the {os.path.dirname(path_lenet)}"
assert os.path.exists(path_tools), f"{path_tools} not exist, please place common_tools.py in the {os.path.dirname(path_tools)}"

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

set_seed()
rmb_label = {"1": 0, "100": 1}

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 16
LR = 0.01
log_interval = 10
val_interval = 1
MOMENTUM = 0.9

# step 1/5: data preparation
split_dir = os.path.join(BASE_DIR, "..", "data", "rmb_split")
if not os.path.exists(split_dir):
    raise Exception(r"data {} not exist, go back to split_dataset.py to generate data".format(split_dir))
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# add the data augmentation: transforms.RandomGrayscale(p=0.9)
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomGrayscale(p=0.9),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# build MyDataset instance
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# build DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# step 2/5: model
net = LeNet(classes=2)
net.initialize_weights()

# step 3/5: loss function
criterion = nn.CrossEntropyLoss()

# step 4/5: optimizer
# select optimizer
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
# set up the learning rate reducing strategy
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# step 5/5: training
train_curve, valid_curve = [], []

for epoch in range(MAX_EPOCH):
    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()
    for i, data in enumerate(train_loader):

        # forward
        inputs, labels = data
        outputs = net(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # count classification prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()

        # append every batch's mean loss, loss.item()
        loss_mean += loss.item()
        train_curve.append(loss.item())
        # print training information in every 10 batch steps, loss_mean is the mean loss in 10 batches
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
            loss_mean = 0.

    scheduler.step()  # 更新学习率

    # validate the model in every epoch
    if (epoch+1) % val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().numpy()

                # every batch's valid mean loss
                loss_val += loss.item()

            # every epoch's valid mean loss
            loss_val_epoch = loss_val / len(valid_loader)
            valid_curve.append(loss_val_epoch)
            # valid_curve.append(loss.item())    # 20191022改，记录整个epoch样本的loss，注意要取平均
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val_epoch, correct_val / total_val))


train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)

# since train_curve record every batch's loss, but train_curve recoder every epoch's loss
# so we need to amplify and interpolate more points between the valid point
valid_x = np.arange(1, len(valid_curve)+1) * train_iters * val_interval
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.show()

# ============================ inference ============================

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# test_dir = os.path.join(BASE_DIR, "test_data")
#
# test_data = RMBDataset(data_dir=test_dir, transform=valid_transform)
# valid_loader = DataLoader(dataset=test_data, batch_size=1)
#
# for i, data in enumerate(valid_loader):
#     # forward
#     inputs, labels = data
#     outputs = net(inputs)
#     _, predicted = torch.max(outputs.data, 1)
#
#     rmb = 1 if predicted.numpy()[0] == 0 else 100
#     print("model predict {} yuan".format(rmb))

#     img_tensor = inputs[0, ...]  # C H W
#     img = transform_invert(img_tensor, train_transform)
#     plt.imshow(img)
#     plt.title("LeNet got {} Yuan".format(rmb))
#     plt.show()
#     plt.pause(0.5)
#     plt.close()

