# -*- coding: utf-8 -*-
"""
@File:        models.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 17, 2024
@Description: DDPM
"""
import os
import sys
import random
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from fusion_unet import FusionUnet


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def gather_(consts, t):
    """
    用于从一个张量consts中根据另一个张量t作为索引  按照指定维度进行取值，并且reshape成一个四维的张量
    :param consts:
    :param t:
    :return:
    """
    return consts.gather(-1, t).reshape(-1, 1, 1, 1)


class DenoiseDiffusion(nn.Module):
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather_(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather_(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.eps_model(xt, t)
        alpha_bar = gather_(self.alpha_bar, t)
        alpha = gather_(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather_(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** .5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t)
        return F.mse_loss(noise, eps_theta)


class Train:
    def __init__(self, args):
        seed_everything(args.seed)
        self.device = torch.device(f"{args.device}" if torch.cuda.is_available() else "cpu")

        # gen samples
        self.n_samples = args.n_samples
        self.images_size = args.images_size
        self.n_steps = args.n_steps
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.step_size = args.step_size
        self.gamma = args.gamma

        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((self.images_size, self.images_size)),  # resize
            transforms.ToTensor()  # 0~1
        ])

        # define dataloader
        # load dataset
        self.dataset = datasets.MNIST(root=args.mnist_path, train=True, download=False, transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        for images, _ in self.dataloader:
            # 'images' will contain the batch of images
            # 'labels' will contain the corresponding labels
            break  # Break after the first batch to inspect its shape
        num_channels = images.shape[1]
        # define model
        self.backbone_model = FusionUnet(in_channels=num_channels, out_channels=num_channels).to(self.device)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.backbone_model,
            n_steps=self.n_steps,
            device=self.device
        )

        # define optimizer
        self.optimizer = optim.Adam(self.diffusion.parameters(), lr=self.lr)

        # define scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def sample(self):
        # 禁用梯度计算
        with torch.no_grad():
            # 生成多个noise
            x = torch.randn([self.n_samples, 1, self.images_size, self.images_size], device=self.device)

            # 对图像进行去噪
            for t_ in range(self.n_steps):
                t = self.n_steps - t_ - 1

                # 在当前时间对图像进行去噪
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))
            return x

    def train(self):
        total_loss = 0

        # 遍历数据集
        for data, _ in self.dataloader:
            # reshape
            data = data.reshape(-1, 1, self.images_size, self.images_size)
            data = data.to(self.device)

            self.optimizer.zero_grad()
            loss = self.diffusion.loss(data)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()
        return total_loss

    def run(self, epochs):
        for epoch in tqdm(range(epochs), file=sys.stdout):
            loss = self.train()

            if epoch % 20 == 0:
                tqdm.write(f"Epoch: {epoch} Loss: {loss}")

    from torchvision.models import resnet18


def get_args():
    import argparse
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="train DDPM.")

    # 添加命令行参数
    parser.add_argument('--mnist_path', type=str, default='/Users/gene/project/tools/llm_tools/data/MNIST_data',
                        help="MNIST_data path")
    parser.add_argument('--device', type=str, default='cuda:0', help="ie. cuda:0 cuda:01 cuda:0123")
    parser.add_argument('--seed', type=int, default=10033, help="random seed")
    parser.add_argument('--epochs', type=int, default=50, help='epoch')
    parser.add_argument('--n_samples', type=int, default=16, help='n_samples')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--images_size', type=int, default=32)
    parser.add_argument('--n_steps', type=int, default=1000, help='n_steps')
    parser.add_argument('--batch_size', type=int, default=8, help='The batch size')
    parser.add_argument('--step_size', type=int, default=10, help='step_size')
    parser.add_argument('--gamma', type=float, default=0.8, help='gamma rate')
    # 解析命令行参数
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train = Train(args)
    train.run(args.epochs)
