# -*- coding: utf-8 -*-
"""
# @file name  : train_lenet.py
# @author     : Jianhua Ma
# @date       : 20211113
# @brief      : RMB classification model training in pytorch lighting
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
from model.lenet import LeNet, LeNet_PLModule
from tools.my_dataset import RMBDataset, RMBDataModule
from tools.common_tools import set_seed

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

path_lenet = os.path.join(BASE_DIR, "..", "model", "lenet.py")
path_tools = os.path.join(BASE_DIR, "..", "tools", "common_tools.py")

assert os.path.exists(path_lenet), f"{path_lenet} not exist, please place lenet.py in the {os.path.dirname(path_lenet)}"
assert os.path.exists(path_tools), f"{path_tools} not exist, please place common_tools.py in the {os.path.dirname(path_tools)}"

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

set_seed()
rmb_label = {"1": 0, "100": 1}

MAX_EPOCH = 10
BATCH_SIZE = 16
LR = 0.01
log_interval = 10
val_interval = 1
MOMENTUM = 0.9

seed_everything(42)

# data preparation
split_dir = os.path.join(BASE_DIR, "..", "data", "rmb_split")
if not os.path.exists(split_dir):
    raise Exception(r"data {} not exist, go back to split_dataset.py to generate data".format(split_dir))
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")


# model
net = LeNet(classes=2)

# loss function
criterion = nn.CrossEntropyLoss()

data = RMBDataModule(
    train_dir=train_dir,
    valid_dir=valid_dir,
    num_workers=4,
    batch_size=BATCH_SIZE
)
data.setup()

model = LeNet_PLModule(
    model=net,
    loss_func=criterion,
    lr=LR
)

early_stop_callback = EarlyStopping(monitor="valid_loss")
checkpoint_callback = ModelCheckpoint(
    monitor='valid_loss',
    dirpath='/Users/jianhuama/Dropbox/deepshareAI/DeepAI_Pytorch_camp/lec06_dataloader',
    filename="sample-RMB-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode='min',
)
trainer = Trainer(
    callbacks=[early_stop_callback, checkpoint_callback],
    gpus=1 if torch.cuda.is_available() else 0,
    max_epochs=MAX_EPOCH,
    deterministic=True
)

# fit the model
trainer.fit(model, data)




