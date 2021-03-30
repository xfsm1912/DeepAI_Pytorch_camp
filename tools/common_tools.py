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

