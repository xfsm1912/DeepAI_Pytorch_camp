# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : Jianhua Ma
# @date       : 20210329
# @brief      : common tool functions
"""
import torch
import random
import psutil
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def transform_invert(img_, transform_train):
    """
    implement inverse transform to the transformed image
    :param img_: tensor
    :param transform_train:
    :return:
    """
    if "Normalize" in str(transform_train):
        # reverse normalization: img*sigma + mean
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    # C*H*W --> W*H*C --> H*W*C
    img_ = img_.transpose(0, 2).transpose(0, 1)
    if "ToTensor" in str(transform_train) or img_.max() < 1:
        img_ = img_.detach().numpy() * 255

    if img_.shape[2] == 3:
        # convert to RGB
        img_ = Image.fromarray(img_.astype("uint8")).convert("RGB")
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype("uint8").squeeze())
    else:
        raise Exception(f"Invalid img shape, expected 1 or 3 in axis 2, but got {img_.shape[2]}!")

    return img_

