# -*- coding: utf-8 -*-
"""
@File:        train.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 22, 2024
@Description: train ddpm model
"""
import os
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

from dataset import load_train_mnist_dataset, load_val_mnist_dataset, show_tensor_image
from fusion_unet import FusionUnet
from models import DenoiseDiffusion


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class Train:
    def __init__(self, args):
        self.device = torch.device(f"{args.device}" if torch.cuda.is_available() else "cpu")
        self.epochs = args.epochs
        self.n_samples = args.n_samples
        self.images_size = args.images_size
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.step_size = args.step_size
        self.gamma = args.gamma
        # 1.define datasets
        # train
        train_data = load_train_mnist_dataset(self.images_size, args.mnist_path)
        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        # val
        val_data = load_val_mnist_dataset(self.images_size, args.mnist_path)
        self.val_dataloader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, drop_last=False)

        # 2.define model
        self.model = DenoiseDiffusion(FusionUnet(in_channels=args.im_channel, out_channels=args.im_channel),
                                      n_steps=args.timesteps, device=self.device).to(self.device)

        # 3.define loss
        self.loss = torch.nn.MSELoss()

        # 4.define optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=args.momentum)  # sgd
        if args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  # adam
        # define scheduler # Reduces LR by a factor of 0.1 every 10 epochs
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
        # Reduces LR by a factor of 0.1 when [10] epochs validation loss never reduce.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.gamma,
                                                                    patience=self.step_size)

        # save
        self.best_model_path = args.best_model_path

    def train(self):
        self.model.train()
        running_loss = 0.0
        for step, data in enumerate(tqdm(self.train_dataloader)):
            self.optimizer.zero_grad()
            loss = self.model.loss(data[0].to(self.device))
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_dataloader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data, _ in tqdm(self.val_dataloader):
                self.optimizer.zero_grad()
                loss = self.model.loss(data[0].to(self.device))
                running_loss += loss.item()
        return running_loss / len(self.val_dataloader)

    def sample(self):
        # load model
        state_dict = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        # 禁用梯度计算
        with torch.no_grad():
            # 生成多个noise
            x = torch.randn([self.n_samples, 1, self.images_size, self.images_size], device=self.device)

            # 对图像进行去噪
            for t_ in range(self.step_size):
                t = self.step_size - t_ - 1

                # 在当前时间对图像进行去噪
                x = self.model.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))
            show_tensor_image(x)
            return x

    def run(self):
        # Initialize best loss
        best_val_loss = float('inf')
        # 5.train
        for epoch in tqdm(range(self.epochs)):
            train_loss = self.train()
            val_loss = self.validate()
            # Get the current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            self.scheduler.step(val_loss)
            # 保存验证集上性能最好的模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f'Saved best model with val loss: {best_val_loss:.4f}')
            print(
                f'Epoch {epoch + 1}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Learning Rate: {current_lr} | Best Loss: {best_val_loss:.4f}')


def get_args():
    import argparse
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="train DDPM.")

    # 添加命令行参数
    parser.add_argument('--mnist_path', type=str, default='../data',
                        help="MNIST_data path")
    parser.add_argument('--best_model_path', type=str, default='./model.pth', help="ie. cuda:0 cuda:01 cuda:0123")
    parser.add_argument('--device', type=str, default='cuda:0', help="ie. cuda:0 cuda:01 cuda:0123")
    parser.add_argument('--seed', type=int, default=10033, help="random seed")
    parser.add_argument('--im_channel', type=int, default=1, help='image channel. ie. rgb=3 gray=1 default->1')
    parser.add_argument('--epochs', type=int, default=100, help='epoch')
    parser.add_argument('--n_samples', type=int, default=16, help='test model n_samples')
    #
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='sgd learning rate momentum')
    parser.add_argument('--images_size', type=int, default=32)
    # diffusion 300 step forward add noise
    parser.add_argument('--timesteps', type=int, default=1000, help='time step')
    parser.add_argument('--batch_size', type=int, default=256, help='The batch size')
    # Reduces LR by a factor of 0.1 every 10 epochs
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--step_size', type=int, default=10, help='step_size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma rate')
    # 解析命令行参数
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train = Train(args)
    # train.run()  # train model
    x = train.model.generate((3, 1, 32, 32), 1000)
    show_tensor_image(x)
