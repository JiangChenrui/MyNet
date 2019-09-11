# 车牌分类

## 文件说明

    data 存储train.txt和test.txt文件的文件夹
    compute_mean 计算图像均值
    train.py 训练
    test.py 测试
    img_select.py 读取xlsx表格中的标签信息
    select_brand_roi.py 从图片中扣除车牌
    brand_classer.py 将车牌图片按类型分类存放
    test_image.py 使用训练好的模型对检测出车牌的车辆图片进行分类

img_mean
[0.46309134, 0.46395576, 0.36762613]
img_std
[0.26067975, 0.24779406, 0.24456058]

## 数据集

共10类

    0 大型汽车号牌 4804
    1 大型新能源汽车号牌 1807
    2 挂车号牌 248
    3 教练车号牌 4691
    4 警用汽车号牌 3617
    5 其他号牌 17
    6 武警号牌 1
    7 香港出入境车牌 1
    8 小型汽车号牌 159
    9 小型新能源号牌 5935

总共21280张图片
现在训练集有20000张，测试集1280张

## 2019/9/3

98 epoch
test set Accuracy:98.20%
avg time: 0.00900s

## 2019/9/4

在数据加载中加入transforms.ColorJitter(0.05, 0.05, 0.05)，对图像的亮度、对比度和饱和度进行随机变化
学习率设置为每20个epoch下降为原来的1/2
69epoch
test set Accuracy:99.14%
avg time: 0.00890s

设置transforms.ColorJitter(0.1, 0.1, 0.1)
test set Accuracy:99.38%
avg time: 0.00814s

## 2019/9/11

训练集 140000
测试集 12877
