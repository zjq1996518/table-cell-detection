import torch
from torch import nn

from model.base_model import BaseModel, ConvBlock, UpConv


class UNet(BaseModel):

    def __init__(self, out_ch=1):
        super(UNet, self).__init__(name='unet')

        n1 = 8
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(3, filters[0])
        self.conv2 = ConvBlock(filters[0], filters[1])
        self.conv3 = ConvBlock(filters[1], filters[2])
        self.conv4 = ConvBlock(filters[2], filters[3])
        self.conv5 = ConvBlock(filters[3], filters[4])

        self.up5 = UpConv(filters[4], filters[3])
        self.up_conv5 = ConvBlock(2 * filters[3], filters[3])

        self.up4 = UpConv(filters[3], filters[2])
        self.up_conv4 = ConvBlock(2 * filters[2], filters[2])

        self.up3 = UpConv(filters[2], filters[1])
        self.up_conv3 = ConvBlock(2 * filters[1], filters[1])

        self.up2 = UpConv(filters[1], filters[0])
        self.up_conv2 = ConvBlock(2 * filters[0], filters[0])

        self.conv = nn.Conv2d(filters[0], out_ch, 1, 1)

    def forward(self, x):
        e1 = self.conv1(x)

        e2 = self.max_pool1(e1)
        e2 = self.conv2(e2)

        e3 = self.max_pool2(e2)
        e3 = self.conv3(e3)

        e4 = self.max_pool3(e3)
        e4 = self.conv4(e4)

        e5 = self.max_pool4(e4)
        e5 = self.conv5(e5)

        d5 = self.up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        out = torch.sigmoid(self.conv(d2))
        return out
