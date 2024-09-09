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
            noise = torch.randn_like(x0, device=x0.device)
        
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t)
        return F.mse_loss(noise, eps_theta)
