# -*- coding: utf-8 -*-
"""
# @file name  : 3_save_checkpoint.py
# @author     : Jianhua Ma
# @date       : 20210403
# @brief      : simulate the accident break
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
import sys

from model.lenet import LeNet
from tools.my_dataset import RMBDataset
from tools.common_tools import set_seed
import torchvision

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

set_seed(1)
rmb_label = {"1": 0, "100": 1}

checkpoint_interval = 5
MAX_EPOCH = 10
BATCH_SIZE = 16
LR = 0.01
log_interval = 10
val_interval = 1


# step 1/5: data preparation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
split_dir = os.path.join(BASE_DIR, "..", "data", "rmb_split")
if not os.path.exists(split_dir):
    raise Exception(r"data {} not exist, go back to split_dataset.py to generate data".format(split_dir))
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomGrayscale(p=0.8),
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
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

# step 4.5/5: reload the model from last checkpoint
path_checkpoint = "./checkpoint_4_epoch.pkl"
checkpoint = torch.load(path_checkpoint)

net.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"]

scheduler.last_epoch = start_epoch

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

        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()

        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
            loss_mean = 0.

    scheduler.step()

    if (epoch+1) % checkpoint_interval == 0:

        checkpoint = {"model_state_dict": net.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "loss": loss,
                      "epoch": epoch}
        path_checkpoint = f"./checkpoint_{epoch}_epoch.pkl"
        torch.save(checkpoint, path_checkpoint)

    # if epoch > 5:
    #     print("训练意外中断...")
    #     break

    # validate the model
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
