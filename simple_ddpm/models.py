# -*- coding: utf-8 -*-
"""
@File:        models.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 23, 2024
@Description:
"""
import torch
import torch.nn as nn

num_steps = 100

beta = torch.linspace(0.0001, 0.02, num_steps)  # T时刻内所有噪声强度
print(beta)
alpha = 1. - beta  # 原图保有率
alpha_bar = torch.cumprod(alpha, dim=0)  # 前N项乘积
print(alpha_bar)
alpha_bar_list = []
for i in range(len(alpha)):
    alpha_bar_i = torch.prod(alpha[:i + 1])
    alpha_bar_list.append(alpha_bar_i)
alpha_bars = torch.tensor(alpha_bar_list, dtype=torch.float16)
print(alpha_bars)
print(alpha_bar - alpha_bars)

alpha_bar_p = torch.cat([torch.Tensor([1]).float(), alpha_bar[:-1]], 0)
alpha_bar_sqrt = torch.sqrt(alpha_bar)
b = torch.tensor([[1, 2, 3], [4, 5, 6]])
cumprod_b_dim0 = torch.cumprod(b, dim=0)
cumprod_b_dim1 = torch.cumprod(b, dim=1)

print(cumprod_b_dim0)
# 输出: tensor([[ 1,  2,  3],
#               [ 4, 10, 18]])

print(cumprod_b_dim1)
# 输出: tensor([[  1,   2,   6],
#               [  4,  20, 120]])
n_steps = n_steps
sigma2 = self.beta


class MLPDiffusion(nn.Module):
    def __init__(self, n_step, num_units=128):
        super(MLPDiffusion, self).__init__()
        self.linear = nn.ModuleList([
            nn.Linear(2, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, 2)
        ])
        self.step_embeddings = nn.ModuleList([
            nn.Embedding(n_step, num_units),
            nn.Embedding(n_step, num_units),
            nn.Embedding(n_step, num_units),
        ])

    def forward(self, x_0, t):
        x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)
        return x
