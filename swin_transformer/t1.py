# -*- coding: utf-8 -*-
"""
@File:        t1.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 10, 2024
@Description:
"""
import torch

x = [i for i in range(1, 4*4+1)]
print(x)
x = torch.Tensor(x)
x = x.view(1, 4, 4, 1)
print(x)
shifted_x = torch.roll(x, shifts=(-1, -1), dims=(1, 2))
print(shifted_x)

x = shifted_x.view(4, 4)
print(x)


shifted_x = torch.roll(shifted_x, shifts=(1, 1), dims=(1, 2))
x = shifted_x.view(4, 4)
print(x)