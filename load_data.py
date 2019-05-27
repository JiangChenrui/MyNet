import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):

    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            # line = line.restrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, lable = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, lable

    def __len__(self):
        return len(self.imgs)


class DataLoad:
    """
    加载数据集
    """

    def __init__(self, batch_size, train_dir, val_dir):
        """
        输入batch_size, train_dir, val_dir
        """
        self.train_dir = train_dir
        self.x_train = None
        self.y_train = None
        self.img_mean = None
        self.train_data_len = 0

        self.val_dir = val_dir
        self.x_val = None
        self.y_val = None
        self.val_data_len = None

        self.x_test = None
        self.y_test = None
        self.test_data_len = 0

        self.batch_size = batch_size

    def load_data(self):
        """
        输出h,w,c,train_num,val_num
        """

        self.x_train, self.y_train = load_path(self.train_dir)
        self.x_val, self.y_val = load_path(self.val_dir)
        self.train_data_len = len(self.x_train)
        self.val_data_len = len(self.x_val)

        img_height = 224
        img_wight = 224
        num_channels = 3
        return img_height, img_wight, num_channels, self.train_data_len, self.val_data_len


def path_to_data(path, size=None):
    if size is not None:
        return cv2.resize(cv2.imread(path), size)
    else:
        return cv2.resize(cv2.imread(path), (224, 224))


def load_path(img_path):
    _path = []
    lables = []
    with open(img_path, 'r') as file:
        # 将文本文件的数据读入到path_data
        path_data = file.readlines()
        for i in range(0, len(path_data)):
            # 将字符串格式转换为正常格式
            path_data[i] = path_data[i].split()
            _path.append(path_data[i][0])
            lables.append(int(path_data[i][1]))
    x = np.array(_path)
    y = np.array(lables)
    return x, y
