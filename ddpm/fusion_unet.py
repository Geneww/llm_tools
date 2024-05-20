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
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel, time_embed_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_embed_dim, out_channel)

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()

    def forward(self, x, t):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # [batch, 64, 112, 112]
        time_emb = self.time_mlp(t)
        time_emb = self.relu(time_emb)  # [batch, 64]
        x = x + time_emb
        return


class FusionUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=[64, 128, 256, 512, 1024], time_embed_dim=32):
        super().__init__()
        # time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(),
            nn.Linear(),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=channels[0], kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=channels[3], out_channels=channels[4], kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=channels[4], out_channels=channels[5], kernel_size=3, padding=1)


if __name__ == '__main__':
    t_emb = SinusoidalPositionEmbeddings(32)
    d = DownSample(3, 64, 32)
    input_x = torch.ones((1, 3, 224, 224))
    input_t = torch.Tensor([1])
    input_t = t_emb(input_t)
    input_a = t_emb(torch.Tensor([1000]))

    d(input_x, input_t)
