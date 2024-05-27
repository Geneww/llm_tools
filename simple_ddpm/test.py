# -*- coding: utf-8 -*-
"""
@File:        test.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 27, 2024
@Description:
"""
import torch

from models import MLPDiffusion

model_path = ""
device = ""

model = MLPDiffusion()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

