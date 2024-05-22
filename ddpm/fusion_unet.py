# -*- coding: utf-8 -*-
"""
@File:        fusion_unet.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 20, 2024
@Description:
"""
import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :].to(time.device)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel, time_embed_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_embed_dim, out_channel)

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x, t):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # [batch, 64, 224, 224]
        time_emb = self.time_mlp(t)
        time_emb = self.relu(time_emb)  # [batch, 64]
        time_emb = time_emb.view(-1, time_emb.shape[1], 1, 1)  # [batch, 64, 1, 1]
        x = x + time_emb  # [batch, 64, 224, 224]
        # conv 2
        x = self.conv2(x)  # [batch, 64, 224, 224]
        x = self.bn2(x)  # [batch, 64, 224, 224]
        x = self.relu(x)  # [batch, 64, 224, 224]
        # down sample
        x = self.conv3(x)  # [batch, 64, 112, 112]
        return x


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel, time_embed_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_embed_dim, out_channel)

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.ConvTranspose2d(out_channel, out_channel, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x, t):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # [batch, 64, 64, 64]
        time_emb = self.time_mlp(t)
        time_emb = self.relu(time_emb)  # [batch, 64]
        time_emb = time_emb.view(-1, time_emb.shape[1], 1, 1)  # [batch, 64, 1, 1]
        x = x + time_emb  # [batch, 64, 64, 64
        # conv 2
        x = self.conv2(x)  # [batch, 64, 64, 64]
        x = self.bn2(x)  # [batch, 64, 64, 64]
        x = self.relu(x)  # [batch, 64, 64, 64]
        # up sample
        x = self.conv3(x)  # [batch, 64, 112, 112]
        return x


class FusionUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=[64, 128, 256, 512, 1024], time_embed_dim=32):
        super().__init__()
        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=channels[0], kernel_size=3, padding=1)
        # down sample
        self.d_conv1 = DownSample(in_channel=channels[0], out_channel=channels[1], time_embed_dim=time_embed_dim)
        self.d_conv2 = DownSample(in_channel=channels[1], out_channel=channels[2], time_embed_dim=time_embed_dim)
        self.d_conv3 = DownSample(in_channel=channels[2], out_channel=channels[3], time_embed_dim=time_embed_dim)
        self.d_conv4 = DownSample(in_channel=channels[3], out_channel=channels[4], time_embed_dim=time_embed_dim)
        # up sample
        self.u_conv4 = UpSample(in_channel=channels[4], out_channel=channels[3], time_embed_dim=time_embed_dim)
        self.u_conv3 = UpSample(in_channel=channels[3], out_channel=channels[2], time_embed_dim=time_embed_dim)
        self.u_conv2 = UpSample(in_channel=channels[2], out_channel=channels[1], time_embed_dim=time_embed_dim)
        self.u_conv1 = UpSample(in_channel=channels[1], out_channel=channels[0], time_embed_dim=time_embed_dim)
        # output 1x1 conv
        self.output = nn.Conv2d(in_channels=channels[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_mlp(t)         # [batch, 32]
        x = self.conv0(x)            # [batch, 64,  224, 224]
        x1 = self.d_conv1(x, t)      # [batch, 128, 112, 112]
        x2 = self.d_conv2(x1, t)     # [batch, 256,  56,  56]
        x3 = self.d_conv3(x2, t)     # [batch, 512,  28,  28]
        x4 = self.d_conv4(x3, t)     # [batch, 1024, 14,  14]
        x = self.u_conv4(x4, t)      # [batch, 512,  28,  28]
        x = self.u_conv3(x + x3, t)  # [batch, 256,  56,  56]
        x = self.u_conv2(x + x2, t)  # [batch, 128, 112, 112]
        x = self.u_conv1(x + x1, t)  # [batch, 64,  224, 224]
        return self.output(x)        # [batch,  3,  224, 224]


if __name__ == '__main__':
    t_emb = SinusoidalPositionEmbeddings(32)
    d = UpSample(3, 64, 32)
    input_x = torch.ones((1, 3, 224, 224))
    # input_t = torch.Tensor([1])
    # input_t = t_emb(input_t)
    # input_a = t_emb(torch.Tensor([1000]))
    # input_v = input_a - input_t
    # number = input_v[input_v > 0]
    # d(input_x, input_t)

    fu = FusionUnet()
    out = fu(input_x, torch.Tensor([1]))
    print(out.shape)
