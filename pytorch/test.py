import torch
import sys
import os
import glob
import numpy as np
import time
import shutil

from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import confusion_matrix
import torchvision.models as torch_models
import models.model as M
from models.MobileNetV3 import MobileNetV3
from models.ShuffleNet_V2 import ShuffleNetV2
from models.InceptionV4 import InceptionV4
# from models.residual_attention_network import ResidualAttentionModel_56
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
cuda_gpu = True


def test(model, img_txt, transforms):
    all_time = 0.0
    acc_num = 0
    fr = open(img_txt, 'r')
    img_paths = fr.readlines()
    y_true = []
    y_pred = []
    for line in img_paths:
        _img_path, label = line.split()
        y_true.append(int(label))
        img_path = os.path.join('/home/jiangchenrui/MyNet', _img_path)
        img = Image.open(img_path).convert('RGB')
        # 将读取的图片转换为4维的张量
        if cuda_gpu:
            input = testTransform(img).cuda().unsqueeze(0)
        else:
            input = testTransform(img).unsqueeze(0)
        start = time.time()
        output = model(input)
        end = time.time()
        all_time += end - start
        _, predicted = torch.max(output.data, 1)
        predict = predicted.detach().cpu().numpy()[0]
        y_pred.append(predict)
        # print(predict)
        if predict == int(label):
            acc_num += 1
        else:
            name = img_path.split('/')
            # print(name[-1])
            shutil.copy(img_path, os.path.join('/home/jiangchenrui/MyNet/data/error_img/' + str(predict), name[-1]))
    print("acc is {:.4f}%".format(acc_num / len(img_paths) * 100))
    print("mean time is {:.6f}ms".format(all_time / len(img_paths) * 1000))
    cm = confusion_matrix(y_true, y_pred)
    print(cm)


if __name__ == '__main__':
    brand_labels = ['no_daocha', 'no_zhengxian', 'yes_daocha', 'yes_zhengxian']
    model_path = '/home/jiangchenrui/MyNet/Result/ResAttention/09-18_09-52-38/best_model_params.pth'
    # model = EfficientNet.from_name('efficientnet-b0', override_params={'num_classes': 4})
    # model = M.MobileNet(num_classes=4)
    # model = M.MobileNetV2(num_classes=4)
    # model = ShuffleNetV2(num_classes=4)
    # model = MobileNetV3(num_classes=4)
    # model = ResidualAttentionModel_56(num_classes=4)
    # model = M.DwresNet(num_classes=4)
    # model = M.DwresNet1_0(num_classes=4)
    # model = M.DwresNet1_1(num_classes=4)
    # model = M.DwAttentionNet(num_classes=4)
    # model = M.DwAttentionNetV2_1(num_classes=4)
    # model = M.DwAttentionNetV2_2(num_classes=4)
    # model = M.DwAttentionNetV2_3(num_classes=4)
    # model = torch_models.resnet50(num_classes=4)
    # model = InceptionV4(num_classes=4)
    # model = M.DwAttentionNetV2(num_classes=4)
    # model = torch_models.vgg16_bn(pretrained=False, num_classes=4)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if cuda_gpu:
        model.cuda()
    normMean = [0.20715627, 0.20715627, 0.20715627]
    normStd = [0.19816825, 0.19816825, 0.19816825]
    normTransform = transforms.Normalize(normMean, normStd)
    testTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normTransform
    ])
    img_txt0 = "../data/test_old"
    print("oldGX2 test")
    test(model, img_txt0, testTransform)
    img_txt1 = '../data/test_newGX2'
    print('\n' + "newGX2 test")
    test(model, img_txt1, testTransform)
    # img_txt2 = '../data/test_newGX3'
    # print("\n" + "newGX3 test")
    # test(model, img_txt2, testTransform)
