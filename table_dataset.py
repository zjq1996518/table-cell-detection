import os
import random

import torch
from torch.utils.data import Dataset
from glob import glob
import cv2
import utils
import numpy as np

from constant import MEAN, STD


def allocation_dataset(data_path, resize_width=512, resize_height=128, allocation_rate=0.8):
    path = glob(f'{data_path}/*.jpg')
    data_paths = []
    mask_paths = []
    for p in path:
        if os.path.basename(p).find('-mask') == -1:
            data_paths.append(p)
            ext = p.find('.jpg')
            mask_path = p[:ext] + '-mask.jpg'
            mask_paths.append(mask_path)

    data_length = len(data_paths)
    index = round(data_length * allocation_rate)

    train_dataset = TableDataset(data_paths[:index], mask_paths[:index], resize_width, resize_height)
    val_dataset = TableDataset(data_paths[index:], mask_paths[index:], resize_width, resize_height)

    return train_dataset, val_dataset


class TableDataset(Dataset):
    def __init__(self, data_paths, mask_paths, resize_width, resize_height):
        """
        :param data_paths: 图片路径列表
        :param mask_paths: mask路径列表
        :param resize_width: 图片resize宽度
        :param resize_height: 图片resize高度
        """

        self.data_paths = data_paths
        self.mask_paths = mask_paths

        self.resize_width = resize_width
        self.resize_height = resize_height

    def __getitem__(self, item):
        table_img = cv2.imread(self.data_paths[item])
        table_img = cv2.cvtColor(table_img, cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(self.mask_paths[item])
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        table_img, mask_img = self.data_augmentation(table_img, mask_img)

        # 所有图片经过3次腐蚀，加强线特征
        table_img = cv2.erode(table_img, np.ones([3, 3], dtype=np.uint8), iterations=3)
        mask_img = cv2.erode(mask_img, np.ones([3, 3], dtype=np.uint8), iterations=3)

        table_img, _, _ = self.standard_resize(table_img, self.resize_height, self.resize_width)
        mask_img, _, _ = self.standard_resize(mask_img, self.resize_height, self.resize_width)

        table_img = (table_img - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
        x = torch.tensor(table_img).float()
        label = torch.tensor(mask_img).float()
        x = x.permute([2, 0, 1])
        label = label/255

        x, _, _ = utils.pad(x, 512, 128)
        label, _, _ = utils.pad(label, 512, 128)

        return x, label

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def standard_resize(img, resize_height, resize_width):
        """
        不改变图片宽高比，对图片进行缩放
        :param img:
        :param resize_height:
        :param resize_width:
        :return:
        """
        # 填充留到pytorch_tensor 部分
        h, w = img.shape[:2]
        ratio_w, ratio_h = 1, 1
        # 两个都大, 图片缩放按照差距较大的那个标准，然后对差距较大的做填充
        if h > resize_height and w > resize_width:
            flag = 0 if resize_height/h < resize_width/w else 1
            scale_h = resize_height if flag == 0 else round(resize_width/w * h)
            scale_w = round(resize_height/h * w) if flag == 0 else resize_width
            img = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_AREA)
            ratio_h = scale_h / h
            ratio_w = scale_w / w
        # 两个都小，图片填充
        elif w <= resize_width and h <= resize_height:
            pass

        # 其中一个大，其中一个小, 对大的那个做缩放，小的做填充
        else:
            flag = 0 if h - resize_height > w - resize_width else 1
            scale_h = resize_height if flag == 0 else round(resize_width/w * h)
            scale_w = round(scale_h/h * w) if flag == 0 else resize_width
            img = cv2.resize(img, (scale_w, scale_h), interpolation=cv2.INTER_AREA)
            ratio_h = scale_h / h
            ratio_w = scale_w / w

        return img, ratio_w, ratio_h

    @staticmethod
    def data_augmentation(img, mask):

        # 通道互换
        random.randint(0, 1)
        if random == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 五个尺度随机拉伸
        rd1 = random.randint(0, 4)
        rd2 = random.randint(0, 4)
        scale1 = 0.8 + rd1 / 10
        scale2 = 0.8 + rd2 / 10
        height, width, _ = img.shape
        height = int(height * scale1)
        width = int(width * scale2)
        # interpolation=cv2.INTER_AREA 如果用默认最近邻插值，细小的单元框特征容易缩放丢失
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)

        # 随机噪声
        rate = random.randint(0, 3) / 10
        noise_count = int(rate * height * width)
        utils.sp_noise(img, noise_count)

        # 随机翻转
        rd = random.randint(-1, 2)
        if rd != 2:
            img = cv2.flip(img, rd)
            mask = cv2.flip(mask, rd)

        height, width, _ = img.shape
        # # 随机小角度旋转
        rotate = random.randint(0, 2)
        mat = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), rotate, 1)
        img = cv2.warpAffine(img, mat, (width, height))
        mask = cv2.warpAffine(mask, mat, (width, height))

        return img, mask
