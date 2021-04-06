# -*- coding: utf-8 -*-
"""
# @file name  : resnet_inference.py
# @author     : Jianhua Ma
# @date       : 20210404
# @brief      : inference demo
"""

import os
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config
vis = True
# vis = False
vis_row = 4

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

classes = ["ants", "bees"]


def img_transform(img_rgb, transform=None):
    """
    convert image from rgb to transformed tensor
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("There is no transform！Please add necessary transform methods")

    img_t = transform(img_rgb)
    return img_t


def get_img_name(img_dir, format="jpg"):
    """

    :param img_dir:
    :param format:
    :return:
    """
    file_names = os.listdir(img_dir)
    img_names = list(filter(lambda x: x.endswith(format), file_names))

    if len(img_names) < 1:
        raise ValueError(f"In the {img_dir}, no file in {format} format.")
    return img_names


def get_model(m_path, vis_model=False):

    resnet18 = models.resnet18()
    num_ftrs = resnet18.fc.in_features
    # replace the fully-connected layer with different classes output size
    resnet18.fc = nn.Linear(num_ftrs, 2)

    checkpoint = torch.load(m_path)
    resnet18.load_state_dict(checkpoint['model_state_dict'])

    if vis_model:
        from torchsummary import summary
        summary(resnet18, input_size=(3, 224, 224), device="cpu")

    return resnet18


if __name__ == "__main__":
    BASEDIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(BASEDIR, "..", "data", "hymenoptera_data"))
    if not os.path.exists(data_dir):
        raise Exception(f"\n{data_dir} not exist，please download finetune_resnet18-5c106cde.pth, place it in "
                        f"\n{os.path.dirname(data_dir)}.")

    img_dir = os.path.join(data_dir, "val", "bees")
    model_path = os.path.abspath(os.path.join(BASEDIR, "..", "data", "resnet_checkpoint_14_epoch.pkl"))
    if not os.path.exists(model_path):
        raise Exception(f"\n{model_path} not exist，please download resnet_checkpoint_14_epoch.pkl  place it in "
                        f"\n{os.path.dirname(model_path)}")

    time_total = 0
    img_list, img_pred = [], []

    # step 1/5: data
    img_names = get_img_name(img_dir)
    num_img = len(img_names)

    # step 2/5: select model, set up evaluation mode
    resnet18 = get_model(model_path, True)
    resnet18.to(device)
    resnet18.eval()

    # step 3/5: prediction
    # we are doing inference, so in evaluation mode, using no_grad() to reduce memory.
    with torch.no_grad():
        for idx, img_name in enumerate(img_names):

            path_img = os.path.join(img_dir, img_name)

            # path --> img
            img_rgb = Image.open(path_img).convert("RGB")

            # img --> tensor
            img_tensor = img_transform(img_rgb=img_rgb, transform=inference_transform)
            img_tensor.unsqueeze_(0)
            img_tensor = img_tensor.to(device)

            # tensor --> output vector
            time_tic = time.time()
            outputs = resnet18(img_tensor)
            time_toc = time.time()

            # visualization
            _, pred_int = torch.max(outputs.data, 1)
            pred_str = classes[int(pred_int)]

            if vis:
                img_list.append(img_rgb)
                img_pred.append(pred_str)

                # plot every 16 figures
                if (idx + 1) % (vis_row * vis_row) == 0 or num_img == idx + 1:
                    for i in range(len(img_list)):
                        plt.subplot(vis_row, vis_row, i + 1).imshow(img_list[i])
                        plt.title("predict:{}".format(img_pred[i]))
                    plt.show()
                    plt.close()
                    img_list, img_pred = [], []

            time_s = time_toc - time_tic
            time_total += time_s

            print('{:d}/{:d}: {} {:.3f}s '.format(idx + 1, num_img, img_name, time_s))

        print("\ndevice:{} total time:{:.1f}s mean:{:.3f}s".
              format(device, time_total, time_total / num_img))
        if torch.cuda.is_available():
            print("GPU name:{}".format(torch.cuda.get_device_name()))

