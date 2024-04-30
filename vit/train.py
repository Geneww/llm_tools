# -*- coding: utf-8 -*-
"""
@File:        train.py
@Author:      Gene
@Software:    PyCharm
@Time:        04月 30, 2024
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

from models import Vit
from datasets import train_loader, test_loader


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
    model = Vit(images_size=args.images_size, num_classes=args.num_classes, embedding_dim=args.embedding_dim,
                patch_size=args.patch_size, dropout=args.dropout, n_head=args.n_head, num_encoders=args.num_encoders).to(device)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    # optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer = SGD(model.parameters(), lr=args.lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)
            # one hot
            label = torch.eye(args.num_classes)[label].long()
            # Zero the gradients
            optimizer.zero_grad()
            # forward
            output = model(data)
            # calculate loss
            loss = criterion(output, label)
            # backward calculate
            loss.backward()
            # update weights
            optimizer.step()
            # Update the scheduler
            scheduler.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)
                label = torch.eye(args.num_classes)[label].long()

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(test_loader)
                epoch_val_loss += val_loss / len(test_loader)

        print(
            f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - test_loss : {epoch_val_loss:.4f} - test_acc: {epoch_val_accuracy:.4f}\n"
        )


def get_args():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="train VIT.")

    # 添加命令行参数
    parser.add_argument('--seed', type=int, default=10033, help="random seed")
    parser.add_argument('--epochs', type=int, default=50, help='epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--device', type=str, default='0', help='devices ie. 0,1,2')
    parser.add_argument('--images_size', type=tuple, default=(28, 28, 1))
    parser.add_argument('--num_classes', type=int, default=10, help='classes type')
    parser.add_argument('--patch_size', type=int, default=7, help='The patch size')
    parser.add_argument('--embedding_dim', type=int, default=16, help='embedding dim ie. patch number **2')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--n_head', type=int, default=4, help='multi head number')
    parser.add_argument('--num_encoders', type=int, default=4, help='encoder layers number')

    # 解析命令行参数
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train(args)
