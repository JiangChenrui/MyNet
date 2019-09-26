import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.autograd import Variable
import numpy as np


def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class ResidualBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_channels, output_channels // 4, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            output_channels // 4,
            output_channels // 4,
            3,
            stride,
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            output_channels // 4, output_channels, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(
            input_channels, output_channels, 1, stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out


class AttentionModule_stage1(nn.Module):
    # input size is 56*56
    def __init__(self,
                 in_channels,
                 out_channels,
                 size1=(56, 56),
                 size2=(28, 28),
                 size3=(14, 14)):
        super(AttentionModule_stage1, self).__init__()
        self.first_residual_blocks = conv_dw(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            conv_dw(in_channels, out_channels),
            conv_dw(in_channels, out_channels))

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax1_blocks = conv_dw(in_channels, out_channels)

        self.skip1_connection_residual_block = conv_dw(in_channels,
                                                       out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax2_blocks = conv_dw(in_channels, out_channels)

        self.skip2_connection_residual_block = conv_dw(in_channels,
                                                       out_channels)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax3_blocks = nn.Sequential(
            conv_dw(in_channels, out_channels),
            conv_dw(in_channels, out_channels))

        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)

        self.softmax4_blocks = conv_dw(in_channels, out_channels)

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.softmax5_blocks = conv_dw(in_channels, out_channels)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=1,
                stride=1, bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=1, stride=1,
                bias=False), nn.Sigmoid())

        self.last_blocks = conv_dw(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(
            out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(
            out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        #
        out_interp3 = self.interpolation3(out_softmax3) + out_softmax2
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4) + out_softmax1
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5) + out_trunk
        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = (1 + out_softmax6) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage2(nn.Module):
    # input image size is 28*28
    def __init__(self,
                 in_channels,
                 out_channels,
                 size1=(28, 28),
                 size2=(14, 14)):
        super(AttentionModule_stage2, self).__init__()
        self.first_residual_blocks = conv_dw(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            conv_dw(in_channels, out_channels),
            conv_dw(in_channels, out_channels))

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax1_blocks = conv_dw(in_channels, out_channels)

        self.skip1_connection_residual_block = conv_dw(in_channels,
                                                       out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax2_blocks = nn.Sequential(
            conv_dw(in_channels, out_channels),
            conv_dw(in_channels, out_channels))

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.softmax3_blocks = conv_dw(in_channels, out_channels)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax4_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=1,
                stride=1, bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=1, stride=1,
                bias=False), nn.Sigmoid())

        self.last_blocks = conv_dw(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(
            out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)

        out_interp2 = self.interpolation2(out_softmax2) + out_softmax1
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out = out_interp2 + out_skip1_connection
        out_softmax3 = self.softmax3_blocks(out)
        out_interp1 = self.interpolation1(out_softmax3) + out_trunk
        out_softmax4 = self.softmax4_blocks(out_interp1)
        out = (1 + out_softmax4) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage3(nn.Module):
    # input image size is 14*14
    def __init__(self, in_channels, out_channels, size1=(14, 14)):
        super(AttentionModule_stage3, self).__init__()
        self.first_residual_blocks = conv_dw(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            conv_dw(in_channels, out_channels),
            conv_dw(in_channels, out_channels))

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            conv_dw(in_channels, out_channels),
            conv_dw(in_channels, out_channels))

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax2_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=1,
                stride=1, bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=1, stride=1,
                bias=False), nn.Sigmoid())

        self.last_blocks = conv_dw(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)

        out_interp1 = self.interpolation1(out_softmax1) + out_trunk
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = (1 + out_softmax2) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


if __name__ == '__main__':
    res_attention1 = AttentionModule_stage1(128, 128)
    input1 = torch.randn(1, 128, 56, 56)
    out_stage1 = res_attention1(input)
    res_attention1 = AttentionModule_stage2(256, 256)
    input2 = torch.randn(1, 256, 28, 28)
