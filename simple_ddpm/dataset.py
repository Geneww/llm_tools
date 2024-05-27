# -*- coding: utf-8 -*-
"""
@File:        dataset.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 23, 2024
@Description:
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_s_curve

import torch

s_curve, _ = make_s_curve(10 ** 4, noise=0.1)
s_curve = s_curve[:, [0, 2]] / 10.0

print(np.shape(s_curve))
data = s_curve.T

fig, ax = plt.subplots()
ax.scatter(*data, color='red', edgecolor='white')
ax.axis('off')

dataset = torch.Tensor(s_curve).float()

plt.show()