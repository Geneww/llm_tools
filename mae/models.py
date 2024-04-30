# -*- coding: utf-8 -*-
"""
@File:        models.py
@Author:      Gene
@Software:    PyCharm
@Time:        04月 30, 2024
@Description:
"""
import torch
import torch.nn as nn


class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim,
                 masking_ratio: float = 0.75,
                 decoder_depth: int = 1,
                 decoder_heads: int = 8,
                 decoder_dim_head: int = 64):
        super(MAE, self).__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        self.encoder = encoder
