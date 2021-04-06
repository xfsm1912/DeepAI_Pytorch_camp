# -*- coding: utf-8 -*-
"""
# @file name  : gan_demo.py
# @author     : Jianhua Ma
# @date       : 20210405
# @brief      : gan training
"""

import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import imageio
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

import sys

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".." + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

from tools.common_tools import set_seed
from torch.utils.data import DataLoader
from tools.my_dataset import CelebADataset
from tools.dcgan import Discriminator, Generator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(1)

data_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "img_align_celeba_2k"))
if not os.path.exists(data_dir):
    raise Exception(f"\n{data_dir} not exist, please download img_align_celeba_2k in \n{os.path.dirname(data_dir)}")

out_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "results", "log_gan"))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print(f"\ncreate folder {out_dir} to record log and training info\n")

ngpu = 0  # Number of GPUs available. Use 0 for CPU mode.
IS_PARALLEL = True if ngpu > 1 else False
checkpoint_interval = 10

image_size = 64
nc = 3
nz = 100
ngf = 64  # 64 / 128
ndf = 64  # 64 / 128
num_epochs = 20
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_idx = 1  # 0.9
fake_idx = 0  # 0.1

lr = 0.0002
batch_size = 64
beta1 = 0.5

d_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -1 ,1
])

if __name__ == "__main__":
    # step 1: data preparation
    train_set = CelebADataset(data_dir=data_dir, transforms=d_transforms)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)

    # show train img
    flag = 0
    # flag = 1
    if flag:
        train_loader_iter = iter(train_loader)
        img_bchw = next(train_loader_iter)
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(img_bchw.to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()
        plt.close()

    # step 2: model
    net_g = Generator(nz=nz, ngf=ngf, nc=nc)
    net_g.initialize_weight()

    net_d = Discriminator(nc=nc, ndf=ndf)
    net_d.initialize_weight()

    net_g.to(device)
    net_d.to(device)

    if IS_PARALLEL and torch.cuda.device_count() > 1:
        net_g = nn.DataParallel(net_g)
        net_d = nn.DataParallel(net_d)

    # step 3: loss
    criterion = nn.BCELoss()

    # step 4: optimizer
    # set up Adam optimizers for both G and D
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))

    # set up the learning rate reducing strategy
    lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=8, gamma=0.1)
    lr_scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=9, gamma=0.1)

    # step 5: iteration
    img_list, g_losses, d_losses, iters = [], [], [], 0

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):

            ############################
            # (1) Update D network
            ###########################

            net_d.zero_grad()

            # training data
            # real img
            real_img = data.to(device)
            b_size = real_img.size()[0]
            real_label = torch.full((b_size,), real_idx, device=device, dtype=torch.float)

            # fake img
            noise = torch.randn(b_size, nz, 1, 1, device=device, dtype=torch.float)
            fake_img = net_g(noise)
            fake_label = torch.full((b_size,), fake_idx, device=device, dtype=torch.float)

            # train D with real img
            out_d_real = net_d(real_img)
            loss_d_real = criterion(out_d_real.view(-1), real_label)

            # train D with fake img
            out_d_fake = net_d(fake_img.detach())
            loss_d_fake = criterion(out_d_fake.view(-1), fake_label)

            # backward
            loss_d_real.backward()
            loss_d_fake.backward()
            loss_d = loss_d_real + loss_d_fake

            # Update D
            optimizer_d.step()

            # record probability
            # D(x)
            d_x = out_d_real.mean().item()
            # D(G(z1))
            d_g_z1 = out_d_fake.mean().item()

            ############################
            # (2) Update G network
            ###########################
            net_g.zero_grad()

            label_for_train_g = real_label
            out_d_fake_2 = net_d(fake_img)

            loss_g = criterion(out_d_fake_2.view(-1), label_for_train_g)
            loss_g.backward()
            optimizer_g.step()

            # record probability
            # D(G(z2))
            d_g_z2 = out_d_fake_2.mean().item()

            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_loader),
                         loss_d.item(), loss_g.item(), d_x, d_g_z1, d_g_z2))

            # Save losses for plotting later
            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())

        # update learning rate
        lr_scheduler_d.step()
        lr_scheduler_g.step()

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = net_g(fixed_noise).detach().cpu()

        img_grid = vutils.make_grid(fake, padding=2, normalize=True).numpy()
        img_grid = np.transpose(img_grid, (1, 2, 0))
        plt.imshow(img_grid)
        plt.title("Epoch:{}".format(epoch))
        # plt.show()
        plt.savefig(os.path.join(out_dir, "{}_epoch.png".format(epoch)))

        # checkpoint
        if (epoch+1) % checkpoint_interval == 0:

            checkpoint = {"g_model_state_dict": net_g.state_dict(),
                          "d_model_state_dict": net_d.state_dict(),
                          "epoch": epoch}
            path_checkpoint = os.path.join(out_dir, "checkpoint_{}_epoch.pkl".format(epoch))
            torch.save(checkpoint, path_checkpoint)

    # plot loss
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(out_dir, "loss.png"))

    # save gif
    imgs_epoch = [int(name.split("_")[0]) for name in
                  list(filter(lambda x: x.endswith("epoch.png"), os.listdir(out_dir)))]
    imgs_epoch = sorted(imgs_epoch)

    imgs = list()
    for i in range(len(imgs_epoch)):
        img_name = os.path.join(out_dir, "{}_epoch.png".format(imgs_epoch[i]))
        imgs.append(imageio.imread(img_name))

    imageio.mimsave(os.path.join(out_dir, "generation_animation.gif"), imgs, fps=2)

    print("done")
