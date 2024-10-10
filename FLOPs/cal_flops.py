# -*- coding: utf-8 -*-
"""
@File:        cal_flops.py
@Author:      Gene
@Software:    PyCharm
@Time:        10月 10, 2024
@Description:
"""
import thop
import torch
import torch.nn as nn
from thop import profile

if __name__ == '__main__':
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3),
    )

    inputs = torch.rand(1, 3, 64, 64)

    MACs, Params = profile(model, inputs=(inputs,), verbose=True)
    FLOPs = MACs * 2

    MACs, FLOPs, Params = thop.clever_format([MACs, FLOPs, Params], "%.3f")
    print(f"MACs: {MACs}")
    print(f"FLOPs: {FLOPs}")
    print(f"Params: {Params}")
