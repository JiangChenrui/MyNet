# -*- coding: utf-8
import torch
import numpy as np
import os
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from load_data import MyDataset
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict
from tensorboardX import SummaryWriter
from efficientnet_pytorch import EfficientNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cuda_gpu = torch.cuda.is_available()
num_classes = 10
model_path = 'Result/efficientnet_pytorch/09-04_10-36-39/best_net_params.pth'
test_dir = 'data/test.txt'

normMean = [0.46309134, 0.46395576, 0.36762613]
normStd = [0.26067975, 0.24779406, 0.24456058]
normTransform = transforms.Normalize(normMean, normStd)
valiTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normTransform,
])
test_data = MyDataset(
    txt_path=test_dir,
    transform=valiTransform,
    image_dir='/data/Det/platewindow_tool/brand_image')
test_loader = DataLoader(dataset=test_data, batch_size=1)

criterion = nn.CrossEntropyLoss().cuda()  # 选择损失函数
# net = efficientnet_b0b((224, 224), num_classes=num_classes)
net = EfficientNet.from_name(
    'efficientnet-b0', override_params={'num_classes': num_classes})
# print(net)

# 加载预训练模型
# net = nn.DataParallel(net)
net.load_state_dict(torch.load(model_path))
net.eval()

loss_sigma = 0.0
cls_num = num_classes
total = 0
correct = 0
time_sum = 0
for i, data in enumerate(test_loader):

    # 获取图片和标签
    inputs, lables = data
    total += lables.size(0)
    if (cuda_gpu):
        inputs, lables = inputs.cuda(), lables.cuda()
        net.cuda()

    inputs, lables = Variable(inputs), Variable(lables)
    # forward
    t_start = time.time()
    outputs = net(inputs)
    t_end = time.time()
    time_sum += t_end - t_start
    outputs.detach()

    # 计算loss
    loss = criterion(outputs, lables)
    loss_sigma += loss.item()

    # 统计
    _, predicted = torch.max(outputs.data, 1)
    correct += torch.sum(predicted == lables).data.cpu().numpy()
    # print(i, t_end - t_start)

print('{} set Accuracy:{:.2%}'.format('test', correct / total))
print('avg time: {:.5f}s'.format(time_sum / total))
