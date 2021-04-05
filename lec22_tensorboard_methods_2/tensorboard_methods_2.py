# -*- coding:utf-8 -*-
"""
@file name  : tensorboard_methods_2.py
@author     : Jianhua Ma
@date       : 20210401
@brief      : tensorboard method 2
"""
import os
import sys
import torch
import time
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tools.my_dataset import RMBDataset
from tools.common_tools import set_seed
from model.lenet import LeNet

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)


# after event files created, run "tensorboard --logdir=./" in the terminal to visualize the data.
set_seed(1)

# example 1: record image
# flag = True
flag = False
if flag:

    writer = SummaryWriter(comment="test_your_comment", filename_suffix="_test_your_filename_suffix")

    # img1 random numbers from normal distribution
    fake_img = torch.randn(3, 512, 512)
    writer.add_image(tag="fake_img", img_tensor=fake_img, global_step=1)
    time.sleep(1)

    # img 2 ones
    fake_img = torch.ones(3, 512, 512)
    time.sleep(1)
    writer.add_image("fake_img", fake_img, 2)

    # img 3 1.1 * ones
    fake_img = torch.ones(3, 512, 512) * 1.1
    time.sleep(1)
    writer.add_image("fake_img", fake_img, 3)

    # img 4 Height * Width
    # random numbers from a uniform distribution on the interval [0, 1)
    fake_img = torch.rand(512, 512)
    writer.add_image("fake_img", fake_img, 4, dataformats="HW")

    # img 5 Height * Width * Channel
    fake_img = torch.rand(512, 512, 3)
    writer.add_image("fake_img", fake_img, 5, dataformats="HWC")

    writer.close()


# example 2: make_grid, make grid figure
flag = True
# flag = False
if flag:
    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    split_dir = os.path.join("..", "data", "rmb_split")
    train_dir = os.path.join(split_dir, "train")
    transform_compose = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor()])
    train_data = RMBDataset(data_dir=train_dir, transform=transform_compose)
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

    # next(iterator), returns the next item from the iterator.
    # here we try to get a single batch from DataLoader
    train_loader_iter = iter(train_loader)
    data_batch, label_batch = next(train_loader_iter)

    # do not combine the two commands as:
    # data_batch, label_batch = next(iter(train_loader))
    # you actually create a new instance of dataloader iterator at each call! Memory is too much

    img_grid = utils.make_grid(tensor=data_batch,
                               nrow=4,
                               normalize=True,
                               scale_each=True)
    # img_grid = utils.make_grid(tensor=data_batch,
    #                            nrow=4,
    #                            normalize=False,
    #                            scale_each=False)

    writer.add_image(tag="input_img", img_tensor=img_grid, global_step=0)
    writer.close()

# example 5: add_graph, visualize module graph
# flag = True
# flag = False
if flag:

    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    # module
    fake_img = torch.randn(1, 3, 32, 32)
    lenet = LeNet(classes=2)

    writer.add_graph(model=lenet, input_to_model=fake_img)
    writer.close()

    from torchsummary import summary
    print(summary(lenet, (3, 32, 32), device="cpu"))
