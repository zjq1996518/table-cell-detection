import torch
from torch import nn

from model.base_model import UpConv, ConvBlock, BaseModel


class NestedUNet(BaseModel):
    def __init__(self, num_classes=1, input_channels=3):
        super().__init__(name='unet++')

        n1 = 8
        nb_filter = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(2, 2)

        self.up1_0 = UpConv(nb_filter[1], nb_filter[1])
        self.up2_0 = UpConv(nb_filter[2], nb_filter[2])
        self.up3_0 = UpConv(nb_filter[3], nb_filter[3])
        self.up4_0 = UpConv(nb_filter[4], nb_filter[4])

        self.up1_1 = UpConv(nb_filter[1], nb_filter[1])
        self.up2_1 = UpConv(nb_filter[2], nb_filter[2])
        self.up3_1 = UpConv(nb_filter[3], nb_filter[3])

        self.up1_2 = UpConv(nb_filter[1], nb_filter[1])
        self.up1_3 = UpConv(nb_filter[1], nb_filter[1])
        self.up2_2 = UpConv(nb_filter[2], nb_filter[2])

        self.conv0_0 = ConvBlock(input_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        self.final1 = nn.Conv2d(nb_filter[0], num_classes, 1)
        self.final2 = nn.Conv2d(nb_filter[0], num_classes, 1)
        self.final3 = nn.Conv2d(nb_filter[0], num_classes, 1)
        self.final4 = nn.Conv2d(nb_filter[0], num_classes, 1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))

        output1 = torch.sigmoid(self.final1(x0_1))
        output2 = torch.sigmoid(self.final2(x0_2))
        output3 = torch.sigmoid(self.final3(x0_3))
        output4 = torch.sigmoid(self.final4(x0_4))

        return output1, output2, output3, output4
