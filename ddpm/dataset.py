# -*- coding: utf-8 -*-
"""
@File:        dataset.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 18, 2024
@Description:
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import numpy as np


def load_train_mnist_dataset(image_size: tuple, mnist_path: str):
    data_transforms = [
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = datasets.MNIST(root=mnist_path, train=True, download=False,
                           transform=data_transform)
    test = datasets.MNIST(root=mnist_path, train=False, download=False,
                          transform=data_transform)
    return torch.utils.data.ConcatDataset([train, test])


def load_val_mnist_dataset(image_size: tuple, mnist_path: str):
    data_transforms = [
        transforms.Resize(image_size),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)
    test = datasets.MNIST(root=mnist_path, train=False, download=False,
                          transform=data_transform)
    return test


def show_tensor_image(images):
    # 数据集和数据加载器
    def remove_invalid_values(tensor):
        array = tensor.numpy()
        # Replace NaN with 0.0, positive infinity with 255, negative infinity with 0
        array = np.nan_to_num(array, nan=0.0, posinf=255, neginf=0)
        # Clamp the values to be within the valid uint8 range
        array = np.clip(array, 0, 255)
        return array.astype(np.uint8)

    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),  # Scale [−1, 1] to [0, 1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),  # Scale [0, 1] to [0, 255]
        transforms.Lambda(lambda t: remove_invalid_values(t)),
        transforms.ToPILImage(),  # Convert to PIL Image
    ])

    num_images = len(images)
    cols = int(np.sqrt(num_images))
    rows = (num_images + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(12, 12))
    axs = axs.flatten() if num_images > 1 else [axs]

    for i, image in enumerate(images):
        img = reverse_transforms(image)
        axs[i].imshow(img)
        axs[i].axis('off')

    for ax in axs[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    img_size = (28, 28)
    mnist_path = "/Users/gene/project/tools/llm_tools/data/MNIST_data"
    data = load_val_mnist_dataset(img_size, mnist_path)
    dataloader = DataLoader(data, batch_size=32, shuffle=True, drop_last=True)
    for i, data in enumerate(dataloader):
        print(i, data[1], data[0].shape)
        show_tensor_image(data[0])
        break
