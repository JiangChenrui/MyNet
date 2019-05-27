import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


vgg16_bn = models.vgg16_bn
AlexNet = models.AlexNet
densenet121 = models.densenet121
SqueezeNet = models.SqueezeNet(num_classes=4)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

    def num_flat_features(self, x):
        size = x.size()
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net_vgg16 = [
    64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 'M', 512, 512,
    512, 'M5', 'FC1', 'FC2', 'FC'
]
net_vgg19 = [
    64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
    'M', 512, 512, 512, 512, 'M5', "FC1", "FC2", "FC"
]


class VGGNet(nn.Module):

    def __init__(self, net_arch, num_classes):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        layers = []
        in_channels = 3
        for arch in net_arch:
            if arch == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif arch == 'M5':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            elif arch == 'FC1':
                layers.append(
                    nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=3,
                        padding=6,
                        dilation=6))
                layers.append(nn.ReLU(inplace=True))
            elif arch == 'FC2':
                layers.append(nn.Conv2d(1024, 1024, kernel_size=1))
                layers.append(nn.ReLU(inplace=True))
            elif arch == 'FC':
                layers.append(nn.Conv2d(1024, self.num_classes, kernel_size=1))
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=arch,
                        kernel_size=3,
                        padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = arch
        self.vgg = nn.ModuleList(layers)

    def forward(self, input_data):
        x = input_data
        for layer in self.vgg:
            x = layer(x)
        out = x
        return out


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.fetures = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    '''
    输入：
        cfg:模型结构
        batch_norm:是否使用batchnorm
    返回：
        模型结构
    '''
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'
    ],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512,
        512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,
        'M', 512, 512, 512, 512, 'M'
    ],
}


def vgg16(**kwargs):
    return VGG(make_layers(cfg['D']), **kwargs)


def vgg16_bn(**kwargs):
    return VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)


print(SqueezeNet)