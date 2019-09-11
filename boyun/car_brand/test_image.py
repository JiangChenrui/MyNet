import torch
import sys
import os
import glob
import numpy as np
import time

from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet


result_dir = '../classification_result'
brand_labels = ['大型汽车后牌', '新能源大车车牌', '挂车车牌', '武警车牌', '不是车牌', '军队车牌', '警车车牌', '大型汽车前牌',
                '快递车牌', '小型汽车车牌', '新能源小车车牌', '教练车牌', '其他车牌']

model_path = 'Result/efficientnet_pytorch/09-04_10-36-39/best_net_params.pth'
model = EfficientNet.from_name('efficientnet-b0', override_params={'num_classes': len(brand_labels)})
model.load_state_dict(torch.load(model_path))
model.eval()
model.cuda()
normMean = [0.46309134, 0.46395576, 0.36762613]
normStd = [0.26067975, 0.24779406, 0.24456058]
normTransform = transforms.Normalize(normMean, normStd)
testTransform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normTransform,
])


def select_roi(img_path, txt_path):
    '''
    从图片中提取检测出的车牌图片并分类后存放
    '''
    img = Image.open(img_path).convert('RGB')
    img_name = img_path.replace('../images/', '').replace('.jpg', '')
    print(img_name, end=' ')
    if not os.path.exists(txt_path):
        return
    labels = np.genfromtxt(txt_path, dtype=int)
    start = time.time()
    if labels.ndim > 1:
        number = labels.shape[0]
        for i in range(1, number):
            label = labels[i][1:]
            crop_img = (label[0] - 10, label[1] - 5, label[2] + 10, label[3] + 5)
            crop = img.crop(crop_img)
            # 使用训练好模型对裁剪车牌进行分类
            predict = brand_classification(crop).detach().cpu().numpy()[0]
            # print(brand_labels[predict])
            crop_path = os.path.join(result_dir, brand_labels[predict])
            crop.save(os.path.join(crop_path, img_name) + '_' + str(i) + '.jpg')
            end = time.time()
    else:
        label = labels[1:]
        crop_img = (label[0] - 10, label[1] - 5, label[2] + 10, label[3] + 5)
        crop = img.crop(crop_img)
        predict = brand_classification(crop).detach().cpu().numpy()[0]
        crop_path = os.path.join(result_dir, brand_labels[predict])
        crop.save(os.path.join(crop_path, img_name) + '.jpg')
        end = time.time()
    print('time is {:.6f}'.format(end - start))


def brand_classification(crop_img):
    '''
    使用训练好的模型对裁剪的车牌进行分类并返回分类结果
    '''
    img = testTransform(crop_img).cuda().unsqueeze(0)   # 图片传入后是3维，需要扩展成4维后才能做输入
    output = model(img)
    _, predicted = torch.max(output.data, 1)
    return predicted.data


if __name__ == '__main__':
    # 图片地址
    image_dir = '../images'

    # 创建分类文件夹
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        for label in brand_labels:
            os.makedirs(os.path.join(result_dir, label))

    # 对车牌进行分类
    for path in glob.glob(os.path.join(image_dir, '*.jpg')):
        label_path = path.replace('images', 'labels').replace('.jpg', '.txt')
        if os.path.exists(label_path):
            select_roi(path, label_path)
