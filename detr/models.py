# -*- coding: utf-8 -*-
"""
@File:        models.py
@Author:      Gene
@Software:    PyCharm
@Time:        04月 30, 2024
@Description: detr net
"""
import torch
import torch.nn as nn

from transformer.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class DETR(nn.Module):
