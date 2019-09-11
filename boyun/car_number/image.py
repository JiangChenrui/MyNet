# !/usr/bin/env python
# coding: utf-8
import random
import os
from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import ImageStat
import cv2


def make_square(im, min_size=0, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, ((size - x) // 2, (size - y) // 2))
    # new_im.show()
    return new_im


def np_make_square(target):
    h, w = target.shape
    if h == w:
        return target
    elif h > w:
        pad_salce = (h - w) // 2
        return np.pad(target, ((0, 0), (pad_salce, pad_salce)), 'constant', constant_values=(0, 0))
    else:
        pad_salce = (w - h) // 2
        return np.pad(target, ((pad_salce, pad_salce), (0, 0)), 'constant', constant_values=(0, 0))


# 图像预处理
def load_data(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    # print(img.size)
    img = make_square(img)
    # print(img.size)
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    # print(target.shape, target.sum())
    target = np_make_square(target)
    # print(target.shape, target.sum())
    '''
    # 宽或高大于2000的resize成768*768
    # '''
    width = img.size[0]
    height = img.size[1]
    # if height >= 800 or width >= 800:
    height_new = 768
    width_new = 768

    a = height / height_new
    b = width / width_new
    s = a * b
    img = img.resize((height_new, width_new), Image.BICUBIC)
    target = cv2.resize(target, (height_new, width_new), interpolation=cv2.INTER_CUBIC) * s
    # print(target.shape, target.sum())

    if False:
        crop_size = (img.size[0] / 2, img.size[1] / 2)
        if random.randint(0, 9) <= -1:

            dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
        else:
            dx = int(random.random() * img.size[0] * 1. / 2)
            dy = int(random.random() * img.size[1] * 1. / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target)  # 镜像
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # target = cv2.resize(
    #     target, (round(target.shape[1] / 8), round(target.shape[0] / 8)),
    #     interpolation=cv2.INTER_CUBIC
    # ) * 64
    target = cv2.resize(target, (384, 384), interpolation=cv2.INTER_CUBIC) * 4
    target = cv2.resize(target, (192, 192), interpolation=cv2.INTER_CUBIC) * 4
    target = cv2.resize(target, (96, 96), interpolation=cv2.INTER_CUBIC) * 4
    # 真值图像的大小缩小了64倍，同时每个点的值扩大了64倍---->总和不变
    # print(target.shape, target.sum())
    target = target[np.newaxis, :]

    return img, target


def selec_ROI(img_path):
    # 原始图像real
    real = cv2.imread(img_path)
    # cv2.namedWindow("real")
    # cv2.imshow("real", real)
    # cv2.waitKey(0)

    # real = np_make_square(real)
    # 前景图像
    # cv2.namedWindow("Image")
    gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("real", gray)
    # cv2.waitKey(0)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 2次腐蚀,3次膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=6)
    img = np_make_square(img)

    # 最后结果reult
    # cv2.imwrite(img_path.replace('images', 'roi3'), img)
    img = cv2.resize(img, (768, 768), interpolation=cv2.INTER_CUBIC) / 255
    return img.astype(int)


if __name__ == "__main__":
    import glob

    image_dir = '/home/boyunvision/data/car_number/data/'
    train_set = os.path.join(image_dir, 'vehicle/test_set', 'images')
    for path in glob.glob(os.path.join(train_set, '*.jpg')):
        # print(path)
        # img, target = load_data(path)
        selec_ROI(path)
        # img = np.array(img)
        # print(img[:, :, 0])
        # img.show()
