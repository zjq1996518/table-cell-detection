import os
from glob import glob

import torch
from torch import nn

from constant import PROJECT_PATH


class BaseModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def save_weight(self, epoch, f1):
        """
        模型存储
        :param epoch:
        :param f1:
        :return:
        """
        device = next(self.parameters()).device
        path = f'{PROJECT_PATH}/weight/{self.name}/'
        model_path = os.path.join(path, f'epoch_{epoch}_f1_{f1}.pth')
        os.makedirs(path, exist_ok=True)
        self.cpu()
        self.eval()
        torch.save(self.state_dict(), model_path)
        latest_path = os.path.join(path, 'latest.pth')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(model_path, latest_path)
        self.to(device)

    def load_weight(self, weight_epoch_name):
        """
        模型加载
        :param weight_epoch_name: 数字，或者latest
        :return:
        """
        path = f'{PROJECT_PATH}/weight/{self.name}/'
        assert os.path.exists(path), '模型不存在'

        latest_model_path = os.path.join(path, 'latest.pth')
        if weight_epoch_name == 'latest':
            assert os.path.exists(latest_model_path), '模型权重加载失败，请检查'
            model_path = latest_model_path

        else:
            model_paths = glob(f'{path}/*.pth')
            model_paths = {model_path.split('_')[1]: model_path for model_path in model_paths if model_path != latest_model_path}
            model_path = model_paths.get(weight_epoch_name)
            assert model_path is not None, '模型权重加载失败，请检查'

        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)


class ConvBlock(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.conv1(x)


class UpConv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, 2, 2),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.InstanceNorm2d(out_ch),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """
    Attention Block
    """

    def __init__(self, g_c, x_c, final_c):
        super().__init__()

        self.w_g = nn.Sequential(
            nn.Conv2d(g_c, final_c, kernel_size=1),
            nn.InstanceNorm2d(final_c)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(x_c, final_c, kernel_size=1),
            nn.InstanceNorm2d(final_c)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(final_c, 1, kernel_size=1),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out
