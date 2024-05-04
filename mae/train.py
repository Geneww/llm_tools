# -*- coding: utf-8 -*-
"""
@File:        train.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 04, 2024
@Description:
"""
from __future__ import print_function

import glob
import os
import random
import argparse
import zipfile
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR

from models import MaskAutoEncoder
from data.datasets import mnist_train_loader, mnist_test_loader


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(args):
    seed_everything(args.seed)
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(device)
    # dataloader

    # define model
    model = MaskAutoEncoder(img_size=args.images_size, patch_size=args.patch_size,
                            embedding_dim=args.embedding_dim,
                            encoder_num=args.encoder_num,
                            num_heads=args.num_heads,
                            masking_ratio=args.masking_ratio,
                            decoder_embedding_dim=args.decoder_embedding_dim,
                            decoder_number=args.decoder_number,
                            decoder_num_heads=args.decoder_num_heads).to(device)

    # optimizer
    # optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer = SGD(model.parameters(), lr=args.lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    for epoch in range(args.epochs):
        epoch_loss = 0

        for data, label in tqdm(mnist_train_loader):
            data = data.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            # forward
            loss, _, _ = model(data)
            # calculate loss
            # backward calculate
            loss.backward()
            # update weights
            optimizer.step()
            # Update the scheduler
            scheduler.step()
            epoch_loss += loss / len(mnist_train_loader)

        with torch.no_grad():
            epoch_val_loss = 0
            for data, label in mnist_test_loader:
                data = data.to(device)
                val_loss, _, _ = model(data)
                epoch_val_loss += val_loss / len(mnist_test_loader)

        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - test_loss : {epoch_val_loss:.4f}\n")


def get_args():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="train VIT.")

    # 添加命令行参数
    parser.add_argument('--seed', type=int, default=10033, help="random seed")
    parser.add_argument('--epochs', type=int, default=50, help='epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--device', type=str, default='0', help='devices ie. 0,1,2')
    parser.add_argument('--images_size', type=tuple, default=(28, 28, 1))
    parser.add_argument('--encoder_num', type=int, default=24, help='classes type')
    parser.add_argument('--decoder_number', type=int, default=8, help='classes type')
    parser.add_argument('--patch_size', type=int, default=7, help='The patch size')
    parser.add_argument('--embedding_dim', type=int, default=16, help='embedding dim ie. patch number **2')
    parser.add_argument('--decoder_embedding_dim', type=int, default=512, help='embedding dim ie. patch number **2')
    parser.add_argument('--masking_ratio', type=float, default=0.75, help='mask rate')
    parser.add_argument('--num_heads', type=int, default=16, help='multi head number')
    parser.add_argument('--decoder_num_heads', type=int, default=16, help='multi head number')
    parser.add_argument('--num_encoders', type=int, default=4, help='encoder layers number')

    # 解析命令行参数
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train(args)
