# -*- coding:utf-8 -*-
"""
@file name  : hook_fmap_vis.py
@author     : Jianhua Ma
@date       : 20210402
@brief      : feature map visualization by hook function
"""
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import torch

from tools.common_tools import set_seed
import torchvision.models as models

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

set_seed(1)  # 设置随机种子

flag = True
# flag = False
if flag:
    with torch.no_grad():
        writer = SummaryWriter(comment="test_your_comment", filename_suffix="_test_your_filename_suffix")

        # data preparation
        path_img = "../data/lena.png"
        normMean = [0.49139968, 0.48215827, 0.44653124]
        normStd = [0.24703233, 0.24348505, 0.26158768]

        norm_transform = transforms.Normalize(normMean, normStd)
        img_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            norm_transform
        ])

        img_pil = Image.open(path_img).convert("RGB")
        if img_transforms is not None:
            img_tensor = img_transforms(img_pil)
        # channel * height * width --> batch * c * h * w
        img_tensor.unsqueeze_(0)

        # Alex model
        alexnet = models.alexnet(pretrained=True)

        # register hook
        fmap_dict = {}
        # Returns an iterator over all modules in the network
        for name, sub_module in alexnet.named_modules():

            if isinstance(sub_module, nn.Conv2d):
                key_name = str(sub_module.weight.shape)
                fmap_dict.setdefault(key_name, [])

                n1, n2 = name.split(".")

                # module: current neural layer
                # input: layer's input
                # output: layer's output, here we consider the output is the feature map, the feature map is appended
                # into the value list in the fmap_dict
                def hook_func(m, i, o):
                    key_name = str(m.weight.shape)
                    fmap_dict[key_name].append(o)

                alexnet._modules[n1]._modules[n2].register_forward_hook(hook_func)

        # forward
        output = alexnet(img_tensor)

        # add image
        for layer_name, fmap_list in fmap_dict.items():
            fmap = fmap_list[0]
            fmap.transpose_(0, 1)

            nrow = int(np.sqrt(fmap.shape[0]))
            fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
            writer.add_image(f"feature map in {layer_name}", fmap_grid, global_step=1000)
            writer.close()
