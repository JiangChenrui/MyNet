import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

SqueeNet = models.SqueezeNet(version=1.1, num_classes=4)
inception = models.inception_v3(num_classes=4)
"""
模型定义
"""


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class MobileNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class MobileNet1_0(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNet1_0, self).__init__()

        # MobileNet0
        # self.model = nn.Sequential(
        #     conv_bn(3, 32, 2),
        #     conv_dw(32, 64, 1),
        #     conv_dw(64, 128, 2),
        #     conv_dw(128, 128, 1),
        #     conv_dw(128, 256, 2),
        #     conv_dw(256, 256, 1),
        #     conv_dw(256, 512, 2),
        #     # conv_dw(512, 512, 1),
        #     # conv_dw(512, 512, 1),
        #     # conv_dw(512, 512, 1),
        #     # conv_dw(512, 512, 1),
        #     # conv_dw(512, 512, 1),
        #     conv_dw(512, 1024, 2),
        #     conv_dw(1024, 1024, 1),
        #     nn.AvgPool2d(7),
        # )

        # MobileNetV1.0
        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 2),
            conv_dw(64, 128, 2),
            conv_dw(128, 256, 2),
            conv_dw(256, 512, 1),
            conv_dw(512, 1024, 2),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


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


# MobileNetV2
# def conv_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


# 带有深度可分离卷积的残差网络
class DwresNet(nn.Module):

    def __init__(self, num_classes=1000, input_size=224, Dropout=0.2):
        super(DwresNet, self).__init__()

        self.conv1 = conv_bn(3, 32, 2)
        self.conv2 = conv_bn(32, 64, 2)

        self.res1_0 = conv_dw(64, 64, 2)
        self.res1_1 = conv_dw(64, 64, 1)

        self.res2_0 = conv_dw(128, 128, 2)
        self.res2_1 = conv_dw(128, 128, 1)

        self.res3_0 = conv_dw(256, 256, 2)
        self.res3_1 = conv_dw(256, 256, 1)

        self.pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x_res1_0 = self.res1_0(x)
        x_res1_1 = x_res1_0
        x_res1_0 = self.res1_1(x_res1_0)
        # 将特征图拼接
        x_res1 = torch.cat((x_res1_0, x_res1_1), 1)

        x_res2_0 = self.res2_0(x_res1)
        x_res2_1 = x_res2_0
        x_res2_0 = self.res2_1(x_res2_0)
        x_res2 = torch.cat((x_res2_0, x_res2_1), 1)

        x_res3_0 = self.res3_0(x_res2)
        x_res3_1 = x_res3_0
        x_res3_0 = self.res3_1(x_res3_0)
        x_res3 = torch.cat((x_res3_0, x_res3_1), 1)

        x_pool = self.pool(x_res3)
        # 将四维数据压缩成2维
        x_pool = x_pool.view(-1, 512)
        result = self.fc(x_pool)

        return result


if __name__ == "__main__":
    import torch
    input = torch.randn(1, 3, 224, 224)
    resnet = models.resnet18(pretrained=False)
    model = DwresNet(num_classes=4)
    output = model(input)
    print(output.shape)
