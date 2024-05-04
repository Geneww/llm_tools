# -*- coding: utf-8 -*-
"""
@File:        models.py
@Author:      Gene
@Software:    PyCharm
@Time:        04月 28, 2024
@Description: vision transformer model file
"""
import torch
import torch.nn as nn

from transformer.transformer import TransformerEncoder, TransformerEncoderLayer

"""
"""


class PatchEmbedding(nn.Module):
    """使用卷积代替切片"""

    def __init__(self, images_size=(224, 224, 3), embedding_dim=768, patch_size=16, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        height, weight, in_channels = images_size
        assert height % patch_size == 0 and weight % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patcher = (height // patch_size) * (weight // patch_size)
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                                 kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        self.cls_token = nn.Parameter(torch.rand(size=(1, 1, embedding_dim)), requires_grad=True)
        # patch * patch + 1 (cls token)
        self.position_embedding = nn.Parameter(torch.rand(size=(1, num_patcher + 1, embedding_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: [n, c, h, w] tensor
        :return: [batch, 197, 768] tensor
        """
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [1, 1, 768] -> [batch, 1, 768]
        x = self.patcher(x)  # [batch, 768, 14, 14]
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2,
                                                       1)  # [batch, 768, 14, 14] -> [batch, 768, 196] -> [batch, 196, 768]

        x = torch.cat([x, cls_token], dim=1)  # [64, 197, 768]
        x += self.position_embedding
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim: int = 768, num_classes: int = 10):
        super(MLP, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.linear = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, x):
        return self.linear(self.layer_norm(x))


class Vit(nn.Module):
    def __init__(self, images_size: tuple = (224, 224, 3), num_classes: int = 10, embedding_dim: int = 768,
                 patch_size: int = 16, dropout: float = 0.1, n_head: int = 12, num_encoders: int = 6):
        super(Vit, self).__init__()
        assert embedding_dim % n_head == 0, "embedding_dim or n_head param error"
        self.patch_embedding = PatchEmbedding(images_size, embedding_dim, patch_size, dropout)
        encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, n_head=n_head, dropout=dropout, dim_feedforward=embedding_dim*4)
        self.encoder_layers = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoders)
        self.mlp = MLP(embedding_dim=embedding_dim, num_classes=num_classes)

    def forward(self, x):
        """
        :param x: [n, c, h, w] tensor
        :return: [n, sql_len, num_classes]
        """
        x = self.patch_embedding(x)
        x = self.encoder_layers(x)  # [batch, 197, 768] -> [768/12=64 8head 197x64] -> [batch, 197, 768]
        x = self.mlp(x)  # [batch, 197, num_classes]
        return x


if __name__ == '__main__':
    from data.datasets import train_loader

    # 224/16 = 14   28/2=14
    vit = Vit((28, 28, 1), num_classes=10, patch_size=7, embedding_dim=4*4*1, n_head=4)

    for image, label in train_loader:
        # n c h w
        x = vit(image)
        print(x.shape)
        break
