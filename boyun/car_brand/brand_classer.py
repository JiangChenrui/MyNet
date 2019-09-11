# -*- coding:utf-8 -*-
import os
import sys
import shutil


""""
对标注好的图片按类别分别存放
"""
brand_txt = 'brand_txt.txt'
images = '/data/Det/platewindow_tool/brand_image'
fr = open(brand_txt, 'r')
image_paths = []
labels = []
for line in fr.readlines():
    line = line.replace('\n', '')
    img_name, label = line.split('\t')
    image_path = os.path.join(images, img_name + '.jpg')
    image_paths.append(image_path)
    labels.append(label)


for i in range(0, len(image_paths)):
    if (not os.path.exists(image_paths[i])) or (labels[i] == ''):
        continue
    if not os.path.exists(os.path.join(images, labels[i])):
        os.makedirs(os.path.join(images, labels[i]))
    shutil.move(image_paths[i], os.path.join(images, labels[i]))
