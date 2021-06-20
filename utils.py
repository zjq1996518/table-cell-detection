import random

import torch
import torch.nn.functional as F


def tensor_pad32m(img, value=0):
    pad_h = 32 - img.shape[-2] % 32 + img.shape[-2]
    pad_w = 32 - img.shape[-1] % 32 + img.shape[-1]

    return pad(img, pad_w, pad_h, value=value)


def pad(img, width, height, value=0):
    pad_w = 0
    pad_h = 0
    pad_width = width - img.shape[-1]
    if pad_width > 0:
        dim1 = pad_width // 2
        dim2 = pad_width // 2 + pad_width % 2
        pad_dim = [dim1, dim2]
        img = F.pad(img, pad_dim, value=value)
        pad_w = dim1

    pad_height = height - img.shape[-2]
    if pad_height > 0:
        dim1 = pad_height // 2
        dim2 = pad_height // 2 + pad_height % 2
        pad_dim = [0, 0, dim1, dim2]
        img = F.pad(img, pad_dim, value=value)
        pad_h = dim1

    return img, pad_w, pad_h


def sp_noise(img, n, color=255, x1=None, y1=None, x2=None, y2=None):
    """
    图片随机添加噪声
    :param img: 原始图片
    :param n: 生成的噪点数量
    :param color: 噪点颜色
    :param x1: 指定图片区域
    :param y1: 同上
    :param x2: 同上
    :param y2: 四个坐标都需要指定
    :return:
    """

    height, width, _ = img.shape

    for _ in range(n):
        if x1 is not None and x2 is not None and y1 is not None and y2 is not None:
            i = random.randint(x1, x2 - 1)
            j = random.randint(y1, y2 - 1)
            img[j, i, :] = color
        else:
            i = random.randint(0, width - 1)
            j = random.randint(0, height - 1)
            img[j, i, :] = color

    return img


def calc_target(y, label):
    with torch.no_grad():
        acc = torch.mean((label == y).float()).item()
        acc_tensor = y[y == label]
        recall = ((torch.sum(acc_tensor == 0)+1e-6) / (torch.sum(y == 0)+1e-6)).item()
        f1 = 2 * (acc * recall) / (recall + acc)

    return acc, recall, f1
