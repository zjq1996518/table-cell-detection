import torch
from torch import nn

from model.base_model import BaseModel, ConvBlock, AttentionBlock, UpConv


class MAUNet(BaseModel):
    def __init__(self, out_ch=1):
        super().__init__(name='ma-unet')

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

        self.att5 = AttentionBlock(filters[3], filters[3], filters[2])
        self.up5 = UpConv(filters[4], filters[3])
        self.up_conv5 = ConvBlock(2 * filters[3], filters[3])

        self.att4 = AttentionBlock(filters[2], filters[2], filters[1])
        self.up4 = UpConv(filters[3], filters[2])
        self.up_conv4 = ConvBlock(2 * filters[2], filters[2])

        self.att3 = AttentionBlock(filters[1], filters[1], filters[0])
        self.up3 = UpConv(filters[2], filters[1])
        self.up_conv3 = ConvBlock(2 * filters[1], filters[1])

        self.att2 = AttentionBlock(filters[0], filters[0], filters[0])
        self.up2 = UpConv(filters[1], filters[0])
        self.up_conv2 = ConvBlock(2 * filters[0], filters[0])

        self.scale_up5 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.scale_up4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.scale_up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.scale_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.beta = torch.nn.Parameter(torch.zeros([1]))

        scale_feature = sum(filters)
        self.conv = nn.Conv2d(scale_feature, scale_feature, 1, 1)

        self.spatial_att_conv = nn.Conv2d(scale_feature, scale_feature, 1, 1)
        self.spatial_conv1 = nn.Conv2d(scale_feature, scale_feature, 1, 1)
        self.spatial_ln = nn.InstanceNorm2d(scale_feature)
        self.spatial_act = nn.LeakyReLU(inplace=True)
        self.spatial_conv2 = nn.Conv2d(scale_feature, scale_feature, 1, 1)

        self.final_conv = nn.Conv2d(2 * sum(filters), out_ch, 1, 1)

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

        d5 = e5

        d4 = self.up5(e5)
        a4 = self.att5(d4, e4)
        d4 = torch.cat((a4, d4), dim=1)
        d4 = self.up_conv5(d4)

        d3 = self.up4(d4)
        a3 = self.att4(d3, e3)
        d3 = torch.cat((a3, d3), dim=1)
        d3 = self.up_conv4(d3)

        d2 = self.up3(d3)
        a2 = self.att3(d2, e2)
        d2 = torch.cat((a2, d2), dim=1)
        d2 = self.up_conv3(d2)

        d1 = self.up2(d2)
        a1 = self.att2(d1, e1)
        d1 = torch.cat((a1, d1), dim=1)
        d1 = self.up_conv2(d1)

        # 多尺度
        feature = torch.cat([d1, self.scale_up2(d2), self.scale_up3(d3), self.scale_up4(d4), self.scale_up5(d5)], dim=1)
        feature = self.conv(feature)
        # 通道 attention
        origin_shape = feature.shape
        feature = feature.reshape(feature.shape[0], feature.shape[1], -1)
        feature_reshape = feature.permute([0, 2, 1])
        att = torch.softmax(torch.bmm(feature, feature_reshape), dim=-1)
        att_channel_feature = torch.bmm(att, feature) * self.beta + feature
        att_channel_feature = att_channel_feature.reshape(*origin_shape)

        # 空间attention
        feature = feature.reshape(*origin_shape)
        att_spatial = self.spatial_att_conv(feature)
        att_spatial = torch.softmax(att_spatial, dim=-1)
        att_spatial = att_spatial * feature
        att_spatial = self.spatial_act(self.spatial_ln(self.spatial_conv1(att_spatial)))
        att_spatial = self.spatial_conv2(att_spatial) + feature

        final_feature = torch.cat([att_spatial, att_channel_feature], dim=1)

        d1 = torch.sigmoid(self.final_conv(final_feature))

        return d1
