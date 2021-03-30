# -*- coding: utf-8 -*-
"""
# @file name  : 1_split_dataset.py
# @author     : Jianhua Ma
# @date       : 2021-03-29
# @brief      : split dataset into training, validation and test set
"""

import os
import random
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    print("the splitted data will be generated in {}\n".format(out_dir))


if __name__ == "__main__":

    dataset_dir = os.path.join(BASE_DIR, "..", "data", "RMB_data")
    split_dir = os.path.join(BASE_DIR, "..", "data", "rmb_split")
    train_dir = os.path.join(BASE_DIR, split_dir, "train")
    valid_dir = os.path.join(BASE_DIR, split_dir, "valid")
    test_dir = os.path.join(BASE_DIR, split_dir, "test")

    if not os.path.exists(dataset_dir):
        raise Exception(f"\n{dataset_dir} not exists, please download RMB_data.rar and place it in "
                        f"{os.path.dirname(dataset_dir)}")

    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    for root, dirs, _ in os.walk(dataset_dir):
        # dirs:['1', '100']
        for sub_dir in dirs:

            imgs = os.listdir(os.path.join(root, sub_dir))
            # filter out the images ended with .jpg.
            # filter(function, iterable)
            imgs = list(filter(lambda x: x.endswith(".jpg"), imgs))

            random.shuffle(imgs)

            img_count = len(imgs)

            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))

            # create three folders: train, valid, test in rmb_split. each folder has subdir "1" and "100"
            for i in range(img_count):
                if i < train_point:
                    # in the train subdirectory, create two subdir named '1' and '100'
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                # here if out_dir not exist, makedir
                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                # copy the img from RMB_data to rmb_split
                shutil.copy(src_path, target_path)

            print('In the Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point - train_point,
                                                                        img_count - valid_point))
