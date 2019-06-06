import random

import cv2
import numpy as np


"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0~255归一至0-1再计算
"""


def compute_mean_std(data_dir, h=224, w=224, CNum=2000):
    """
        输入：
            data_dir: 文本文件位置
            h: 目标图像高
            w：目标图像宽
            CNum：图像数量
        返回：
            means: 均值
            stdevs: 标准差
    """
    means, stdevs = [], []
    imgs = np.zeros([h, w, 3, 1])
    with open(data_dir, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)

        for i in range(CNum):
            img_path = lines[i].split()[0]

            img = cv2.imread(img_path)
            img = cv2.resize(img, (h, w))

            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)
            print(i)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()
    stdevs.reverse()
    return means, stdevs


normMean, normStd = compute_mean_std("data/train_old", 224, 224, 5)
print(normMean, normStd)
