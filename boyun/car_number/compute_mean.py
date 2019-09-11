# encoding:utf-8
import torch
import glob
import os
import cv2
import numpy as np
from torchvision import transforms, utils

# from skimage import io, transform
import dataset


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


def _make_square(img_old):
    img = []
    for i in range(img_old.shape[2]):
        img_channel = img_old[:, :, i]
        img.append(np_make_square(img_channel))
    img = np.array(img)
    img = img.transpose(1, 2, 0)
    return img


def compute_mean_std(image_dir):
    image_path = []
    for img_path in glob.glob(os.path.join(image_dir, '*.jpg')):
        image_path.append(img_path)

    dataloader = torch.utils.data.DataLoader(
        dataset.listDataset(
            image_path,
            shuffle=False,
            transform=transforms.Compose([transforms.ToTensor()])
        ))

    pop_mean = []
    pop_std0 = []
    # print(dataset)
    for i, (img, label) in enumerate(dataloader):
        print(i)
        # print(i, label)
        # shape (batch_size, 3, height, width)
        numpy_image = img.numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))

        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)

    return pop_mean, pop_std0


def compute_mean(path):
    file_names = os.listdir(path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    for file_name in file_names:
        img = cv2.imread(os.path.join(path, file_name), 1)
        img = _make_square(img)
        per_image_Bmean.append(np.mean(img[:, :, 0]))
        per_image_Gmean.append(np.mean(img[:, :, 1]))
        per_image_Rmean.append(np.mean(img[:, :, 2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)

    return R_mean, G_mean, B_mean


def compute_std(path):
    file_names = os.listdir(path)
    per_image_Rstd = []
    per_image_Gstd = []
    per_image_Bstd = []
    for file_name in file_names:
        img = cv2.imread(os.path.join(path, file_name), 1)
        img = _make_square(img)
        per_image_Bstd.append(np.std(img[:, :, 0]))
        per_image_Gstd.append(np.std(img[:, :, 1]))
        per_image_Rstd.append(np.std(img[:, :, 2]))
    R_std = np.std(per_image_Rstd)
    G_std = np.std(per_image_Gstd)
    B_std = np.std(per_image_Bstd)

    return R_std, G_std, B_std


path = "/home/boyunvision/data/car_number/data/vehicle/train_set/images"
# normMean, normStd = compute_mean_std(path)
# print(normMean, normStd)

print(compute_std(path))
