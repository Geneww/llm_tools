# -*- coding: utf-8 -*-
"""
@File:        datasets.py
@Author:      Gene
@Software:    PyCharm
@Time:        04月 30, 2024
@Description:
"""
from PIL import Image
from torchvision import datasets
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0
        return img_transformed, label


# 设置数据转换，将Tensor正规化到[-1,1]之间，并添加通道维度
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载训练集
train_set = datasets.MNIST('/Users/gene/project/tools/llm_tools/data/MNIST_data/', download=True, train=True, transform=transform)
mnist_train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# 下载测试集
test_set = datasets.MNIST('/Users/gene/project/tools/llm_tools/data/MNIST_data/', download=True, train=False, transform=transform)
mnist_test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

