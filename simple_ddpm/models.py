# -*- coding: utf-8 -*-
"""
@File:        models.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 23, 2024
@Description:
"""
import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, n_steps: int = 1000, time_embedding_dim: int = 100):
        super(TimeEmbedding, self).__init__()
        self.n_steps = n_steps
        # 开始位置编码部分,n_steps * time_embedding_dim 的矩阵，即1000 * 100
        # 1000是1000个时刻步数，100是一个时刻用多长的向量来表示，1000*100这个矩阵用于表示一个时刻的信息
        self.time_embedding = torch.zeros(n_steps, time_embedding_dim)
        time = torch.arange(0, n_steps, dtype=torch.float).reshape((n_steps, 1))  # time：[n_steps,1],即[1000,1]
        # 先把括号内的分式求出来,time[1000,1],分母是[50],通过广播机制相乘后是[1000,50]
        div_term = time / pow(10000.0, torch.arange(0, time_embedding_dim, 2).float() / time_embedding_dim)
        # 再取正余弦
        self.time_embedding[:, 0::2] = torch.sin(div_term)
        self.time_embedding[:, 1::2] = torch.cos(div_term)

        # 将pe作为固定参数保存到缓冲区，不会被更新
        self.register_buffer('Time Embedding', self.time_embedding)

    def forward(self, t):
        t = t * self.time_embedding  # [n, 100] * [200, 100]
        return t


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel, time_embed_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
        )

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
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
        )

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
    def __init__(self, in_channels=3, channels=[64, 128, 256, 512, 1024],
                 time_steps=1000, time_embed_dim=100):
        super().__init__()
        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Embedding(time_steps, time_embed_dim),
            TimeEmbedding(time_embed_dim, time_embed_dim),
        )
        self.time_mlp.requires_grad_(False)  # 禁用梯度更新

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
        self.output = nn.Conv2d(in_channels=channels[0], out_channels=in_channels, kernel_size=1)

    def forward(self, x, t):
        t = self.time_mlp(t)  # [batch, 32]
        x = self.conv0(x)  # [batch, 64,  224, 224]
        x1 = self.d_conv1(x, t)  # [batch, 128, 112, 112]
        x2 = self.d_conv2(x1, t)  # [batch, 256,  56,  56]
        x3 = self.d_conv3(x2, t)  # [batch, 512,  28,  28]
        x4 = self.d_conv4(x3, t)  # [batch, 1024, 14,  14]
        x = self.u_conv4(x4, t)  # [batch, 512,  28,  28]
        x = self.u_conv3(x + x3, t)  # [batch, 256,  56,  56]
        x = self.u_conv2(x + x2, t)  # [batch, 128, 112, 112]
        x = self.u_conv1(x + x1, t)  # [batch, 64,  224, 224]
        return self.output(x)  # [batch,  3,  224, 224]


class DDPM(nn.Module):
    def __init__(self, backbone=FusionUnet, in_channels=3, n_steps=200, beta_range=(1e-4, 0.02), device=None):
        super(DDPM, self).__init__()
        self.device = device
        self.backbone = backbone(in_channels, time_steps=n_steps).to(device)
        self.n_steps = n_steps
        self.betas = torch.linspace(*beta_range, n_steps).to(device)  # T时刻内所有噪声强度
        self.alphas = 1. - self.betas  # 原图保有率
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)  # 前N项乘积

    def forward(self, x_0, t, eta=None):
        """add noise"""
        n, c, h, w = x_0.shape
        a_bar = self.alphas_bar[t]
        # 创建噪声
        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)  # noise
        # add noise image
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x_0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def predict_noise(self, x, t):
        """predict noise"""
        noise_p = self.backbone(x, t)
        return noise_p


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm = DDPM(backbone=FusionUnet, in_channels=1, n_steps=200, device=device)

    x_0 = torch.randn((2, 1, 28, 28)).to(device)
    t = torch.randint(0, ddpm.n_steps, (2,)).to(device)

    print(t.shape)

    noise_img = ddpm.forward(x_0, t)
    print(noise_img.shape)
    pred_noise = ddpm.predict_noise(noise_img, t)
    print(pred_noise.shape)
