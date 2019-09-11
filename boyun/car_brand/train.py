# -*- coding: utf-8
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# from compute_mean import compute_mean_std
from load_data import MyDataset, weight_init
from efficientnet_pytorch import EfficientNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append("..")
cuda_gpu = torch.cuda.is_available()
train_dir = 'data/train.txt'
test_dir = 'data/test.txt'

batch_size = 64
lr_init = 1e-2
max_epoch = 200
num_classes = 13

# model
# vgg = models.vgg16_bn(num_classes=num_classes)
net = EfficientNet.from_name(
    'efficientnet-b0', override_params={'num_classes': num_classes})

# 加载之前的模型继续训练
# model_path = 'Result/efficientnet_pytorch/09-10_14-43-03/best_net_params.pth'
# net = nn.DataParallel(net)
# net.load_state_dict(torch.load(model_path))
# print(net)

# 权重初始化
weight_init(net)
if (cuda_gpu):
    net = net.cuda()  # 将模型转为cuda类型

# log
result_dir = 'Result/'

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, 'efficientnet_pytorch', time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir)

# 数据预处理
# normMean, normStd = compute_mean_std(train_dir, 224, 224, 2000)
normMean = [0.46309134, 0.46395576, 0.36762613]
normStd = [0.26067975, 0.24779406, 0.24456058]
normTransform = transforms.Normalize(normMean, normStd)
trainTrainsform = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.RandomCrop(224),
    transforms.ColorJitter(0.15, 0.15, 0.15),
    transforms.ToTensor(), normTransform,
])

# valiTransform = transforms.Compose([transforms.ToTensor(), normTransform])
valiTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normTransform,
])

# 构建MyDataset实例
train_data = MyDataset(
    txt_path=train_dir,
    transform=trainTrainsform,
    image_dir='/data/Det/platewindow_tool/brand_image')
test_data = MyDataset(
    txt_path=test_dir,
    transform=valiTransform,
    image_dir='/data/Det/platewindow_tool/brand_image')

# 构建Dataloder
train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

# 定义损失函数和优化器
# 选择损失函数
criterion = nn.CrossEntropyLoss().cuda()
# 选择优化器
optimizer = optim.SGD(
    net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)
# optimizer = optim.Adam(net.parameters(), lr=lr_init)
# 设置学习率下降策略
# 自适应调整学习率
# scheduler = torch.optim.lr_scheduler.\
#     ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True, min_lr=0, eps=1e-08)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.5)
best_model = 0

#  训练
for epoch in range(max_epoch):
    loss_sigma = 0.0  # 记录每个epoch的loss之和
    correct = 0.0
    total = 0.0
    net.train()
    scheduler.step()  # 更新学习率
    print(scheduler.get_lr()[0])

    for i, data in enumerate(train_loader):
        # 获取图片和标签
        inputs, lables = data

        # 将数据转换为cuda
        if (cuda_gpu):
            inputs, lables = inputs.cuda(), lables.cuda()

        inputs, lables = Variable(inputs), Variable(lables)

        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, lables)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
        total += lables.size(0)
        correct += torch.sum(predicted == lables.data).cpu().numpy()
        loss_sigma += loss.item()

        # 每1000个iteration打印一次训练信息， loss为10个iteration的平均
        if i % 100 == 99:
            loss_avg = loss_sigma / 100
            loss_sigma = 0.0
            print(
                "Time: {:.19s} Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%}"
                .format(
                    str(datetime.now()), epoch + 1, max_epoch, i + 1,
                    len(train_loader), loss_avg, correct / total))

            # 记录训练loss
            writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            # 记录learning rate
            writer.add_scalars('learning rate',
                               {'learning rate': scheduler.get_lr()[0]}, epoch)
            # 记录Accuracy
            writer.add_scalars('Accuracy_group', {'train_acc': correct / total},
                               epoch)

    # 为每个epoch记录梯度，权值
    for name, layer in net.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    # 验证集验证
    if epoch % 1 == 0:
        loss_sigma = 0.0
        net.eval()
        total_test = 0
        correct_test = 0

        for i, data in enumerate(test_loader):
            # 获取图片和标签
            inputs, lables = data
            total_test += lables.size(0)
            if (cuda_gpu):
                inputs, lables = inputs.cuda(), lables.cuda()
                net.cuda()

            inputs, lables = Variable(inputs), Variable(lables)
            # forward
            outputs = net(inputs)
            outputs.detach()

            # 计算loss
            loss = criterion(outputs, lables)
            loss_sigma += loss.item()

            # 统计
            _, predicted = torch.max(outputs.data, 1)
            correct_test += torch.sum(predicted == lables.data).cpu().numpy()
            # print(i, int(correct_test), int(total_test))

        Accuracy = correct_test / total_test

        print('{} set Accuracy:{:.2%}'.format('test', Accuracy))

        # 保存模型
        if Accuracy >= best_model:
            best_model = Accuracy
            print("save model")
            net_save_path = os.path.join(log_dir, 'best_net_params.pth')
            torch.save(net.state_dict(), net_save_path)


print('Finished Training')
