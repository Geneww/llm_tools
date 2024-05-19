# -*- coding: utf-8 -*-
"""
@File:        dataset.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 18, 2024
@Description:
"""

import torch
import torch.utils.data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def show_images(datset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(datset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples / cols) + 1, cols, i + 1)
        plt.imshow(img[0])


print(1e-10 * 3)