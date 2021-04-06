# -*- coding: utf-8 -*-
"""
# @file name  : seg_demo.py
# @author     : Jianhua Ma
# @date       : 20210404
# @brief      : torch.hub make pic segementation by deeplab-V3
"""

import os
import time
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    path_img = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "demo_img3.png"))
    if not os.path.exists(path_img):
        raise Exception("\n{} not exist，please download PortraitDataset and place it in \n{}".format(
            path_img, os.path.dirname(path_img)))

    # transform
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # step 1: load data and download deeplab-V3 model
    input_image = Image.open(path_img).convert("RGB")

    model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
    model.eval()

    # step 2: preprocess
    input_tensor = preprocess(input_image)
    input_bchw = input_tensor.unsqueeze(0)

    # step 3: to device
    if torch.cuda.is_available():
        input_bchw = input_bchw.to(device)
        model.to(device)

    # step 4: forward
    with torch.no_grad():
        tic = time.time()
        print(f"input img tensor shape:{input_bchw.shape}")
        # output_4d shape: [1, 21, 730, 574]
        output_4d = model(input_bchw)["out"]
        # output shape: [21, 730, 574]
        # now, there are 21 matrix corresponding to 21 class predictions. In each 2d matrix, the pixel means the
        # probability belong to the current class. In the 0th matrix some pixels are very high, so they are background
        # In the 12th matrix some pixel are very high, so they are dog
        output = output_4d[0]
        print(f"pass: {(time.time()-tic):3f}s use: {device}")
        print(f"output img tensor shape: {output.shape}")

    # along the class axis dimension, select out the maximum pixel's indice. If 0, it means the max value comes
    # from the 0th matrix and it is background. If 12, it means the max value comes from 12th matrix and it is dog.
    output_predictions = output.argmax(0)

    # step 5 vis
    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)
    plt.subplot(121).imshow(r)
    plt.subplot(122).imshow(input_image)
    plt.show()

    # appendix
    classes = ['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor']



