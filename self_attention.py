# -*- coding: utf-8 -*-
"""
@File:        self_attention.py
@Author:      Gene
@Software:    PyCharm
@Time:        04月 13, 2024
@Description:
"""
import numpy as np

def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.expand_dims(np.sum(t, axis=1), 1)
    return a

query = np.array([
    [1, 0, 2],
    [2, 2, 2],
    [2, 1, 3]
])

key = np.array([
    [0, 1, 1],
    [4, 4, 0],
    [2, 3, 1]
])

value = np.array([
    [1, 2, 3],
    [2, 8, 0],
    [2, 6, 3]
])

score = query @ key.T
print(score)

score = soft_max(score)
print(score)

out = score @ value
print(out)
