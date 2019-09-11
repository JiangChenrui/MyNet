import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    print(pad_beg, pad_end)
    return padded_inputs


class SeparableConv2d(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes,
            inplanes,
            kernel_size,
            stride,
            1,
            dilation,
            groups=inplanes,
            bias=bias)
        self.bn = nn.BatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        # print('0 ', x.size())
        # x = fixed_padding(
        #     x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        # print('1 ', x.size(), '\n')
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


# encoder block
class Block(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 start_with_relu=True):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(
                inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(planes)
        else:
            self.skip = None
        first_conv = []

        rep = []

        # Deep SeparableConv1
        if start_with_relu:
            first_conv.append(nn.ReLU())
            first_conv.append(
                SeparableConv2d(inplanes, planes // 4, 3, 1, dilation))
            first_conv.append(nn.BatchNorm2d(planes // 4))
            first_conv.append(nn.ReLU())
        if not start_with_relu:
            first_conv.append(
                SeparableConv2d(inplanes, planes // 4, 3, 1, dilation))
            first_conv.append(nn.BatchNorm2d(planes // 4))
            first_conv.append(nn.ReLU())

        rep.append(SeparableConv2d(planes // 4, planes // 4, 3, 1, dilation))
        rep.append(nn.BatchNorm2d(planes // 4))

        if stride != 1:
            rep.append(nn.ReLU())
            rep.append(SeparableConv2d(planes // 4, planes, 3, 2))
            rep.append(nn.BatchNorm2d(planes))

        if stride == 1:
            rep.append(nn.ReLU())
            rep.append(SeparableConv2d(planes // 4, planes, 3, 1))
            rep.append(nn.BatchNorm2d(planes))

        self.first_conv = nn.Sequential(*first_conv)
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.first_conv(inp)
        x = self.rep(x)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


class enc(nn.Module):
    """
    encoders:
    stage:stage=X ,where X means encX,example: stage=2 that means you defined the encoder enc2
    """

    def __init__(self, in_channels, out_channels, stage):
        super(enc, self).__init__()
        if (stage == 2 or stage == 4):
            rep_nums = 4
        elif (stage == 3):
            rep_nums = 6
        rep = []
        rep.append(
            Block(in_channels, out_channels, stride=2, start_with_relu=False))
        for i in range(rep_nums - 1):
            rep.append(
                Block(
                    out_channels, out_channels, stride=1, start_with_relu=True))

        self.reps = nn.Sequential(*rep)

    def forward(self, lp):
        x = self.reps(lp)
        return x


class fcattention(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(fcattention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            # nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1000, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # print(y.size())
        y = self.fc(y).view(b, c, 1, 1)
        # print(y.size())
        # y = self.conv(y)
        return x


class xceptionAx3(nn.Module):
    """
    """

    def __init__(self):
        super(xceptionAx3, self).__init__()
        self.seen = 0
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False), nn.BatchNorm2d(num_features=8), nn.ReLU())
        self.enc2a = enc(in_channels=8, out_channels=48, stage=2)
        self.enc2b = enc(in_channels=240, out_channels=48, stage=2)
        self.enc2c = enc(in_channels=240, out_channels=48, stage=2)

        self.enc3a = enc(in_channels=48, out_channels=96, stage=3)
        self.enc3b = enc(in_channels=144, out_channels=96, stage=3)
        self.enc3c = enc(in_channels=144, out_channels=96, stage=3)

        self.enc4a = enc(in_channels=96, out_channels=192, stage=4)
        self.enc4b = enc(in_channels=288, out_channels=192, stage=4)
        self.enc4c = enc(in_channels=288, out_channels=192, stage=4)

        # self.enc2a_to_decoder_dim_reduction = nn.Sequential(
        #     nn.Conv2d(48, 32, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(32), nn.ReLU())
        # self.enc2b_to_decoder_dim_reduction = nn.Sequential(
        #     nn.Conv2d(48, 32, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(32), nn.ReLU())
        # self.enc2c_to_decoder_dim_reduction = nn.Sequential(
        #     nn.Conv2d(48, 32, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(32), nn.ReLU())

        self.fca1_to_decoder_dim_reduction = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU())
        self.fca2_to_decoder_dim_reduction = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU())
        self.fca3_to_decoder_dim_reduction = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU())

        self.merge_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU())
        self.last_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        # backbone stage a
        stage1 = self.conv1(x)
        # print("stage1:",stage1.size())
        stage_enc2a = self.enc2a(stage1)
        # print(stage_enc2a.size())

        stage_enc3a = self.enc3a(stage_enc2a)
        # print('stage_enc3a:',stage_enc3a.size())

        stage_enc4a = self.enc4a(stage_enc3a)
        # print('stage_enc4a:', stage_enc4a.size())

        up_fca1 = F.upsample(
            stage_enc4a,
            stage_enc2a.size()[2:],
            mode='bilinear')

        # print('up_fca1:', up_fca1.size())

        # stage b
        stage_enc2b = self.enc2b(torch.cat((up_fca1, stage_enc2a), 1))
        # print('stage_enc2b:', stage_enc2b.size())
        stage_enc3b = self.enc3b(torch.cat((stage_enc2b, stage_enc3a), 1))
        # print(stage_enc3b.size())
        stage_enc4b = self.enc4b(torch.cat((stage_enc3b, stage_enc4a), 1))
        # up_fca2 = F.interpolate(
        #     stage_enc4b,
        #     stage_enc2b.size()[2:],
        #     mode='bilinear',
        #     align_corners=False)
        up_fca2 = F.upsample(
            stage_enc4b,
            stage_enc2b.size()[2:],
            mode='bilinear')
        # stage c
        stage_enc2c = self.enc2c(torch.cat((up_fca2, stage_enc2b), 1))
        stage_enc3c = self.enc3c(torch.cat((stage_enc2c, stage_enc3b), 1))
        stage_enc4c = self.enc4c(torch.cat((stage_enc3c, stage_enc4b), 1))

        x_fca1 = self.fca1_to_decoder_dim_reduction(stage_enc4a)
        # x_fca1 [1, 32, 48, 48]
        # x_fca1_up = F.upsample(
        #     x_fca1, stage_enc3a.size()[2:], mode='bilinear')
        x_fca2 = self.fca2_to_decoder_dim_reduction(stage_enc4b)
        # x_fca2 [1, 32, 24, 24]
        # x_fca2_up = F.upsample(
        #     x_fca2, stage_enc3a.size()[2:], mode='bilinear')
        x_fca3 = self.fca3_to_decoder_dim_reduction(stage_enc4c)
        # x_fca3 [1, 32, 12, 12]
        x_fca3_up = F.upsample(x_fca3, x_fca2.size()[2:], mode='bilinear')
        x_fca2_add_3 = self.merge_conv(x_fca2 + x_fca3_up)
        x_fca2_add_3_up = F.upsample(x_fca2_add_3, x_fca1.size()[2:], mode='bilinear')
        x_fca_add = self.merge_conv(x_fca1 + x_fca2_add_3_up)

        x_fca_up = F.upsample(x_fca_add, stage_enc3a.size()[2:], mode='bilinear')
        x_merge = self.merge_conv(x_fca_up)
        # print(x_fca_up.size())
        result = self.last_conv(x_merge)
        # print(result.size())
        # result = F.interpolate(
        #     result, x.size()[2:], mode='bilinear', align_corners=False)
        # print(result.size())
        return result

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print("Have initialized modle")


def dfanet(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = xceptionAx3(**kwargs)
    if pretrained:
        print("loading pretrained model.....")
        pretrained_params = torch.load(
            'model_ilsvrc_xeception_0056.pt')['model_state_dict']
        new_state_dict = pretrained_params.copy()
        for key in pretrained_params.keys():
            if key.find('enc2') > 0:
                for x in ['enc2a', 'enc2b', 'enc2c']:
                    new_key = key.replace('enc2', x)
                    new_state_dict[new_key] = pretrained_params[key]
            if not key.find('enc3') > 0:
                for x in ['enc3a', 'enc3b', 'enc3c']:
                    new_key = key.replace('enc3', x)
                    new_state_dict[new_key] = pretrained_params[key]
            if not key.find('enc4') > 0:
                for x in ['enc4a', 'enc4b', 'enc4c']:
                    new_key = key.replace('enc4', x)
                    new_state_dict[new_key] = pretrained_params[key]
        pop_keys = []
        for key in new_state_dict.keys():
            if key.find('reps.0.first_conv.0.') > 0:
                pop_keys.append(key)
            if key.find('.reps.0.skip.weight') > 0:
                pop_keys.append(key)

        for key in pop_keys:
            new_state_dict.pop(key)
        model.load_state_dict(new_state_dict, strict=False)
    return model


if __name__ == "__main__":
    from torch.nn import CrossEntropyLoss

    criterion = CrossEntropyLoss()

    net = dfanet()
    # net=enc(in_channels=8,out_channels=48,stage=2)

    input = torch.randn(1, 3, 768, 768)

    time_start = time.time()
    outputs = net(input)
    time_end = time.time()
    print(str(time_end - time_start))

    # torch.save(net.state_dict(), "model.pth")
    # print(outputs.size())
