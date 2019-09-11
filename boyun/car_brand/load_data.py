# -*- coding: utf-8
import numpy as np
import cv2
import sys
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn


def weight_init(model):
    """
    输入：pytorch网络模型
    对模型参数进行初始化
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    print("Weight has init")


class MyDataset(Dataset):

    def __init__(self,
                 txt_path,
                 transform=None,
                 target_transform=None,
                 image_dir=None):
        fh = open(txt_path, 'r', encoding='utf-8')
        imgs = []
        i = 0
        for line in fh:
            line = line.strip('\n')
            # line = line.restrip()
            # print(i, line)
            i += 1
            if line is None:
                return
            words = line.split()
            imgs.append((words[0], int(words[1])))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.image_dir = image_dir

    def __getitem__(self, index):
        fn, lable = self.imgs[index]
        fn = os.path.join(self.image_dir, fn)
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, lable

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    from torchvision import transforms, datasets
    test_dir = 'data/train.txt'
    normMean = [0.3497724, 0.35888246, 0.37229323]
    normStd = [0.2726704, 0.2739602, 0.2761853]
    normTransform = transforms.Normalize(normMean, normStd)
    valiTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(), normTransform
    ])
    test_data = MyDataset(
        txt_path=test_dir,
        transform=valiTransform,
        image_dir='/data/Det/platewindow_tool/brand_image')
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    for i, data in enumerate(test_loader):
        image, label = data
        print(i, label)
