# encoding:utf-8
import torch
import glob
import os
import cv2
import numpy as np
from torchvision import transforms, utils


def compute_mean(path, h, w):
    fr = open(path, 'r')
    file_path = fr.readlines()
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    for file_name in file_path:
        img_path, _ = file_name.split('\t')
        img = cv2.imread(os.path.join('/home/jiangchenrui/MyNet', img_path), 1)
        img = cv2.resize(img, (h, w))
        img = img.astype(np.float32) / 255.
        per_image_Bmean.append(np.mean(img[:, :, 0]))
        per_image_Gmean.append(np.mean(img[:, :, 1]))
        per_image_Rmean.append(np.mean(img[:, :, 2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)

    return R_mean, G_mean, B_mean


def compute_std(path, h, w):
    # file_names = os.listdir(path)
    fr = open(path, 'r')
    file_path = fr.readlines()
    per_image_Rstd = []
    per_image_Gstd = []
    per_image_Bstd = []
    for file_name in file_path:
        img_path, _ = file_name.split('\t')
        img = cv2.imread(os.path.join('/home/jiangchenrui/MyNet', img_path), 1)
        img = cv2.resize(img, (h, w))
        img = img.astype(np.float32) / 255.
        per_image_Bstd.append(np.std(img[:, :, 0]))
        per_image_Gstd.append(np.std(img[:, :, 1]))
        per_image_Rstd.append(np.std(img[:, :, 2]))
    R_std = np.std(per_image_Rstd)
    G_std = np.std(per_image_Gstd)
    B_std = np.std(per_image_Bstd)

    return R_std, G_std, B_std


img_txt = '../data/train_old'

print("mean is ", compute_mean(img_txt, 224, 224))
print("std is ", compute_std(img_txt, 224, 224))
