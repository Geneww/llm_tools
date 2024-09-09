# -*- coding: utf-8 -*-
"""
@File:        models.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 23, 2024
@Description:
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from fusion_unet import FusionUnet


class DenoiseDiffusion(nn.Module):
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def gather_(self, consts, t):
        return consts.gather(-1, t).reshape(-1, 1, 1, 1)

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.gather_(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - self.gather_(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.eps_model(xt, t)
        alpha_bar = self.gather_(self.alpha_bar, t)
        alpha = self.gather_(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = self.gather_(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** 0.5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0, device=x0.device)

        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t)
        return F.mse_loss(noise, eps_theta)

    def generate(self, shape: Tuple[int], num_steps: int = None):
        if num_steps is None:
            num_steps = self.n_steps

        xt = torch.randn(shape, device=next(self.parameters()).device)
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((shape[0],), t, device=xt.device, dtype=torch.long)
            xt = self.p_sample(xt, t_tensor)
        return xt


if __name__ == '__main__':
    # Example usage
    # Assuming `fusion_unet` is an instance of a U-Net model
    ddpm = DenoiseDiffusion(eps_model=FusionUnet(), n_steps=300, device=torch.device('cpu'))
    generated_images = ddpm.generate(shape=(1, 3, 32, 32))
