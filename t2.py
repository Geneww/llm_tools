# -*- coding: utf-8 -*-
"""
@File:        t2.py
@Author:      Gene
@Software:    PyCharm
@Time:        04月 19, 2024
@Description:
"""
import torch
from torch import nn
from torch import functional as F

import cv2

# Load the image
image = cv2.imread('/Users/gene/project/temp/1.jpeg')  # Replace 'your_image_path.jpg' with the path to your image
image = cv2.resize(image, (224, 224))

# Split the image into 14 patches of size 16x16
patch_size = 16
patches = []
for i in range(0, 224, patch_size):
    print(i, i+patch_size)
    patch = image[i:i+patch_size, i:i+patch_size]
    patches.append(patch)

# # Display or save the patch images
# for idx, patch in enumerate(patches):
#     cv2.imshow(f'Patch {idx+1}', patch)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cls_token = nn.Parameter(torch.rand(size=(1, 1, 4)), requires_grad=True)
ps = nn.Parameter(torch.rand(size=(1, 14, 4)), requires_grad=True)

print(cls_token)
print(ps)

print(1e-1)


import torch

# 创建一个示例张量
input_tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])

# 创建一个索引张量
index_tensor = torch.tensor([[0, 0], [1, 0], [1, 1]])

# 沿着第一个维度收集数据
output = torch.gather(input_tensor, 1, index_tensor)

print(output)

a = [0, 1, 2, 3, 4]

print(a[:-1])
print(a[-1:])