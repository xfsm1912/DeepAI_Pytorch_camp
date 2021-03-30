# -*- coding: utf-8 -*-
"""
# @file name  : dataset.py
# @author     : Jianhua Ma
# @date       : 20210329
# @brief      : define different dataloader class for different dataset
"""

import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {"1": 0, "100": 1}


class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """

        :param data_dir:
        :param transform:
        """
        self.label_name = {"1": 0, "100": 1}
        # data_info store all the pic dirs and labels, load the sample by index in the DataLoader
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert("RGB")   # 0-255

        if self.transform is not None:
            img = self.transform(img)   # here implement the transform, convert into tensor

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        """

        :param data_dir:
        :return: a list of tuple, (path_img, label), path_img is the path of an img
        """
        data_info = []
        for root, dirs, _ in os.walk(data_dir):
            # traverse labels
            for sub_dir in dirs:
                img_names_list = os.listdir(os.path.join(root, sub_dir))
                img_names_list = list(filter(lambda x: x.endswith('.jpg'), img_names_list))

                # traverse img
                for i in range(len(img_names_list)):
                    img_name = img_names_list[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info


