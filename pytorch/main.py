import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as torch_models
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import models.model as M
from load_data import MyDataset
from utils.utils import show_confMat, validate
from models.MobileNetV3 import MobileNetV3
from models.ShuffleNet_V2 import ShuffleNetV2
from models.InceptionV4 import InceptionV4
# from models.residual_attention_network import ResidualAttentionModel_56
from efficientnet_pytorch import EfficientNet

os.environ["CUDA_VISIBLE_DEVICES"] = '4'  # 使用哪几个GPU进行训练


sys.path.append("..")
cuda_gpu = torch.cuda.is_available()

batch_size = 64
lr_init = 1e-2
max_epoch = 60

# model
vgg16_bn = torch_models.vgg16_bn(pretrained=False, num_classes=4)
MobileNet = M.MobileNet(num_classes=4)
# SqueezeNet = models.SqueezeNet(version=1.1, num_classes=4)
inception = InceptionV4(num_classes=4)
MobileNetV2 = M.MobileNetV2(num_classes=4)
MobileNetV3 = MobileNetV3(num_classes=4)
resnet50 = torch_models.resnet50(num_classes=4)
MobileNet1_0 = M.MobileNet1_0(num_classes=4)
ShuffleNetV2 = ShuffleNetV2(num_classes=4)
DwresNet = M.DwresNet(num_classes=4)
DwresNet1_0 = M.DwresNet1_0(num_classes=4)
DwresNet1_1 = M.DwresNet1_1(num_classes=4)
# ResAttetion = ResidualAttentionModel_56(num_classes=4)
DwAttentionNet = M.DwAttentionNet(num_classes=4)
DwAttentionNetV2 = M.DwAttentionNetV2(num_classes=4)
DwAttentionNetV2_1 = M.DwAttentionNetV2_1(num_classes=4)
DwAttentionNetV2_2 = M.DwAttentionNetV2_2(num_classes=4)
DwAttentionNetV2_3 = M.DwAttentionNetV2_3(num_classes=4)
EfficientNet = EfficientNet.from_name('efficientnet-b0', override_params={'num_classes': 4})
net = DwresNet
print(net)

# 权重初始化
M.weight_init(net)
# net._initialize_weights()
if (cuda_gpu):
    net = torch.nn.DataParallel(net).cuda()  # 将模型转为cuda类型
    # net = net.cuda()

train_dir = '../data/train_old'
val_dir = '../data/test_old'

# log
result_dir = 'Result/'

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, 'test', time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir)

classes_name = ['no_daocha', 'no_zhengxian', 'yes_daocha', 'yes_zhengxian']

# 数据预处理
# normMean, normStd = compute_mean_std(train_dir, 224, 224, 2000)
# print(normMean, normStd)
normMean = [0.20715627, 0.20715627, 0.20715627]
normStd = [0.19816825, 0.19816825, 0.19816825]
normTransform = transforms.Normalize(normMean, normStd)
trainTrainsform = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.5, 0.5, 0.5),
    transforms.ToTensor(), normTransform
])

# valiTransform = transforms.Compose([transforms.ToTensor(), normTransform])
valiTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), normTransform
])

# 构建MyDataset实例
train_data = MyDataset(txt_path=train_dir, transform=trainTrainsform)
valid_data = MyDataset(txt_path=val_dir, transform=valiTransform)

# 构建Dataloder
train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=1)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss().cuda()  # 选择损失函数
optimizer = optim.SGD(
    net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)  # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=1, gamma=0.95)  # 设置学习率下降策略

# 训练
best_model = 0

for epoch in range(max_epoch):

    loss_sigma = 0.0  # 记录每个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率
    print(scheduler.get_lr()[0])
    net.train()

    for i, data in enumerate(train_loader):
        # 获取图片和标签
        inputs, lables = data

        # 将数据转换为cuda
        if (cuda_gpu):
            inputs, lables = Variable(inputs.cuda()), Variable(lables.cuda())
        else:
            inputs, lables = Variable(inputs), Variable(lables)

        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, lables)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
        # 将输出数据去cuda，转为numpy
        if (cuda_gpu):
            predicted = predicted.cpu().numpy()
        total += lables.size(0)
        correct += (predicted == lables).squeeze().sum().numpy()
        loss_sigma += loss.item()

        # 每10个iteration打印一次训练信息， loss为10个iteration的平均
        if i % 100 == 99:
            loss_avg = loss_sigma / 100
            loss_sigma = 0.0
            print(
                "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%}"
                .format(epoch + 1, max_epoch, i + 1, len(train_loader),
                        loss_avg, correct / total))

            # 记录训练loss
            writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            # 记录learning rate
            writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
            # 记录Accuracy
            writer.add_scalars('Accuracy_group', {'train_acc': correct / total},
                               epoch)

    # 为每个epoch记录梯度，权值
    for name, layer in net.named_parameters():
        writer.add_histogram(name + '_grad',
                             layer.grad.cpu().numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    # 验证集验证
    if epoch % 2 == 1:
        loss_sigma = 0.0
        cls_num = len(classes_name)
        conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
        net.eval()
        for i, data in enumerate(valid_loader):

            # 获取图片和标签
            inputs, lables = data
            if (cuda_gpu):
                inputs, lables = Variable(inputs.cuda()), Variable(lables.cuda())
            else:
                inputs, lables = Variable(inputs), Variable(lables)
            # forward
            outputs = net(inputs)
            # outputs.detach_()
            outputs.detach()

            # 计算loss
            loss = criterion(outputs, lables)
            loss_sigma += loss.item()

            # 统计
            _, predicted = torch.max(outputs.data, 1)

            # 统计混淆矩阵
            for j in range(len(lables)):
                cate_i = lables[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.0

        Accuracy = conf_mat.trace() / conf_mat.sum()
        print('{} set Accuracy:{:.2%}'.format('Valid',
                                              conf_mat.trace() / conf_mat.sum()))
        # 记录Loss, accuracy
        writer.add_scalars('Loss_group',
                           {'valid_loss': loss_sigma / len(valid_loader)},
                           epoch)
        writer.add_scalars('Accuracy_group',
                           {'valid_acc': conf_mat.trace() / conf_mat.sum()},
                           epoch)

        if Accuracy >= best_model:
            best_model = Accuracy
            net_save_path = os.path.join(log_dir, 'best_model_params.pth')
            torch.save(net.state_dict(), net_save_path)
            print("Save model success")


print('Finished Training')

# 绘制混淆矩阵图
conf_mat_train, train_acc = validate(net, train_loader, 'train', classes_name)
conf_mat_valid, valid_acc = validate(net, valid_loader, 'valid', classes_name)

show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)
