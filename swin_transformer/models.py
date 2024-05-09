# -*- coding: utf-8 -*-
"""
@File:        models.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 08, 2024
@Description:
"""
import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    @staticmethod
    def drop_path(x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(224, 224, 3), patch_size=4, embedding_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.projection = nn.Conv2d(in_channels=img_size[2], out_channels=embedding_dim, kernel_size=patch_size,
                                    stride=patch_size)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        n, c, h, w = x.shape
        assert h == self.img_size[0] and w == self.img_size[
            1], f"Input size ({h}*{w}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.projection(x)  # [batch, 96, 56, 56]
        x = x.flatten(2).transpose(2, 1)  # [batch, 96, 3136] -> [batch, 3136, 96]
        x = self.layer_norm(x)  # [batch, 3136, 96]
        return x


class WindowAttention(nn.Module):
    def __init__(self, num_hidden, window_size, n_head, bias=False, dropout=0.1):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_head = n_head
        self.q_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.k_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.v_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.concat_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, in_dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, dropout=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU,
                 fused_window_process=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        mlp_hidden_dim = int(in_dim * mlp_ratio)

        self.layer_norm_1 = nn.LayerNorm(in_dim)
        # W-MSA/SW-WSA
        self.window_attention = WindowAttention(num_hidden=in_dim, window_size=window_size, n_head=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(in_dim)
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, in_dim),
            nn.Dropout(dropout)
        )

    @staticmethod
    def window_partition(x, window_size):
        n, h, w, c = x.shape  # [batch, 56, 56, 96]
        x = x.view(n, h // window_size, window_size, w // window_size, window_size, c)  # [batch, 8, 7, 8, 7, 96]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)  # [64*batch, 7, 7, 96]
        print(x.shape)
        return x

    @staticmethod
    def window_reverse(windows, window_size, h, w):
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
        return x

    def forward(self, x):
        """x: batch, sql_len, channels"""
        h, w = self.input_resolution
        n, l, c = x.shape
        assert l == h * w, "input feature has wrong size"
        shortcut = x

        # layer norm
        x = self.layer_norm_1(x)  # [batch, 3136, 96]
        x = x.view(n, h, w, c)  # [batch, 56, 56, 96]
        print(x.shape)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # partition windows
        x_windows = self.window_partition(shifted_x, self.window_size)  # [64*batch, 7, 7, 96]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # [64*batch, 49, 96]
        print(x_windows.shape)
        # W-MSA/SW-MSA
        attention_windows = self.window_attention(x_windows)  # [64*batch, 49, 96]
        print(attention_windows.shape)
        # merge windows
        attention_windows = attention_windows.view(-1, self.window_size, self.window_size, c)  # [64*batch, 7, 7, 96]
        print(attention_windows.shape)
        # reverse window
        shifted_x = self.window_reverse(attention_windows, self.window_size, h, w)  # [batch, 56, 56, 96]
        print(shifted_x.shape)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(n, h * w, c)  # [batch, 3136, 96]
        # shortcut 1
        x_ = shortcut + self.drop_path(x)  # [batch, 3136, 96]
        # layer norm
        x = self.layer_norm_2(x_)
        # MLP
        x = self.mlp(x)
        # shortcut 2
        x = x_ + self.drop_path(x)  # [batch, 3136, 96]
        print(x.shape)
        return x


class SwinTransformer(nn.Module):
    r"""
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # patch partition
        self.patch_embedding = PatchEmbedding()

        # swin Transformer block
        self.swin_transformer_block = 1

    def forward(self, x):
        x = self.patch_embedding(x)  # [batch, c, h, w] -> [batch, 3136, 96]
        x = 1
        return


if __name__ == '__main__':
    images = torch.ones((1, 3, 224, 224))
    # print(images)
    # pe = PatchEmbedding()
    # out = pe(images)

    x = torch.ones((2, 3136, 96))
    st = SwinTransformerBlock(96, (56, 56), 12)
    st(x)
