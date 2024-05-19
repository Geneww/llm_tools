# -*- coding: utf-8 -*-
"""
@File:        unet.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 19, 2024
@Description: unet model
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.max_pool(self.conv(x))


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv = DoubleConv(in_channels, out_channels)
        self.up.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias, 0)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diff_x // 2, diff_x - diff_x // 2,
                                    diff_y // 2, diff_y - diff_y // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channel=100,  channels=[64, 128, 256, 512, 1024], dropout=0.5):
        super().__init__()
        # encoder
        self.down1 = DownSample(in_channels, channels[0])
        self.down2 = DownSample(channels[0], channels[1])
        self.down3 = DownSample(channels[1], channels[2])
        self.down4 = DownSample(channels[2], channels[3])
        self.dropout4 = nn.Dropout2d(dropout)
        self.down5 = DownSample(channels[3], channels[4])
        self.dropout5 = nn.Dropout2d(dropout)

        # decoder
        self.up1 = UpSample(channels[4], channels[3])
        self.up2 = UpSample(channels[3], channels[2])
        self.up3 = UpSample(channels[2], channels[1])
        self.up4 = UpSample(channels[1], channels[0])
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(channels[0], channels[0] // 2, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(channels[0] // 2, channels[0], kernel_size=3, padding=1, bias=False),
        )
        # output
        self.out = nn.Conv2d(channels[0], out_channel, kernel_size=1)

    def forward(self, x):
        e1 = self.down1(x)      # [batch, 64, 112, 112]
        e2 = self.down2(e1)     # [batch, 128, 56, 56]
        e3 = self.down3(e2)     # [batch, 256, 28, 28]
        e4 = self.down4(e3)     # [batch, 512, 14, 14]
        e4 = self.dropout4(e4)
        e5 = self.down5(e4)     # [batch, 1024, 7,  7]
        f = self.dropout5(e5)
        d4 = self.up1(f, e4)    # [batch, 512, 14, 14]
        d3 = self.up2(d4, e3)   # [batch, 256, 28, 28]
        d2 = self.up3(d3, e2)   # [batch, 128, 56, 56]
        d1 = self.up4(d2, e1)   # [batch, 64, 112,112]
        d0 = self.up(d1)        # [batch, 64, 224,224]
        out = self.out(d0)      # [batch, num_class,224,224]
        return out


if __name__ == '__main__':
    unet = Unet()
    input_x = torch.ones((1, 3, 256, 256))
    unet(input_x).to("mps")
