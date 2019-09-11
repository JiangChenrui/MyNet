# -*- coding:utf-8 -*-
# 图像提取
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
from PIL import Image


def select_roi(img_path, txt_path):
    img = Image.open(img_path)
    if not os.path.exists(txt_path):
        return
    labels = np.genfromtxt(txt_path, dtype=int)
    # if labels is None:
    #     return
    label = []
    if labels.shape == (2, 5):
        for i in range(0, len(labels)):
            if(labels[i][0] == 1):
                label = labels[i][1:]
    else:
        if(labels[0] == 1):
            label = labels[1:]
        else:
            return
    crop_img = (label[0] - 10, label[1] - 5, label[2] + 10, label[3] + 5)
    print(crop_img)
    crop = img.crop(crop_img)
    # Image._show(crop)
    crop.save(img_path.replace('images', 'brand_image'))


if __name__ == "__main__":
    images = "/data/Det/platewindow_tool/images"
    img_paths = []
    for path in glob.glob(os.path.join(images, '*.jpg')):
        img_paths.append(path)
    for img_path in img_paths:
        print(img_path)
        txt_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        select_roi(img_path, txt_path)
