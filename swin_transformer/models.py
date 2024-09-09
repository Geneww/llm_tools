# -*- coding: utf-8 -*-
"""
@File:        models.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 08, 2024
@Description:
"""
import copy

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
        self.window_size = window_size

        self.q_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.k_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.v_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.concat_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), n_head))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        n, sql_len, c = x.shape
        # 分多个头  [batch, seq_len, hidden_size] -> [batch, seq_len, head_size, d_k] -> [batch, head_size, seq_len, d_k]
        d_k = self.num_hidden // self.num_head
        q = self.q_mat(x).view(x.size(0), -1, self.num_head, d_k).transpose(1, 2)
        k = self.k_mat(x).view(x.size(0), -1, self.num_head, d_k).transpose(1, 2)
        v = self.v_mat(x).view(x.size(0), -1, self.num_head, d_k).transpose(1, 2)
        score = q @ k.transpose(-2, -1)

        # relative position
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # [49, 49, n_head]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [n_head, 49, 49]
        score = score + relative_position_bias.unsqueeze(0)
        # mask
        if mask is not None:
            score = score.view(n // mask.shape[0], mask.shape[0], self.num_head, sql_len, sql_len)
            score = score + mask.unsqueeze(1).unsqueeze(0)
            score = score.view(-1, self.num_head, sql_len, sql_len)  # [batch, n_head, 49, 49]

        score = self.softmax(score)

        score = self.attention_dropout(score)
        attention = score @ v  # [batch, head, sql_len ,d_k]
        x = attention.transpose(1, 2).reshape(n, sql_len, c)
        x = self.concat_mat(x)
        x = self.dropout(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, in_dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4, dropout=0., drop_path=0.,
                 act_layer=nn.GELU):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        mlp_hidden_dim = int(in_dim * mlp_ratio)

        self.layer_norm_1 = nn.LayerNorm(in_dim)
        # W-MSA
        self.window_attention = WindowAttention(num_hidden=in_dim, window_size=window_size, n_head=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(in_dim)
        # MLP
        self.mlp_1 = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, in_dim),
            nn.Dropout(dropout)
        )
        #
        self.layer_norm_3 = nn.LayerNorm(in_dim)
        # SW-WSA
        self.shift_window_attention = WindowAttention(num_hidden=in_dim, window_size=window_size, n_head=num_heads)
        self.layer_norm_4 = nn.LayerNorm(in_dim)
        # MLP2
        self.mlp_2 = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, in_dim),
            nn.Dropout(dropout)
        )

        # attention mask
        # calculate attention mask for SW-MSA
        H, W = self.input_resolution
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = self.window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer("attn_mask", attn_mask)

    @staticmethod
    def window_partition(x, window_size):
        n, h, w, c = x.shape  # [batch, 56, 56, 96]
        x = x.view(n, h // window_size, window_size, w // window_size, window_size, c)  # [batch, 8, 7, 8, 7, 96]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)  # [64*batch, 7, 7, 96]
        return x

    @staticmethod
    def window_reverse(windows, window_size, h, w):
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
        return x

    def forward(self, x):
        """x: batch, sql_len, channels"""
        h, w = self.input_resolution  # [56, 56]
        n, l, c = x.shape  # [batch, 3136, 96]
        assert l == h * w, "input feature has wrong size"

        shortcut = x
        # 1.layer norm 1
        x = self.layer_norm_1(x)  # [batch, 3136, 96]
        x = x.view(n, h, w, c)  # [batch, 56, 56, 96]

        # 2 W-MSA
        # 2.1partition windows
        x_windows = self.window_partition(x, self.window_size)  # [64*batch, 7, 7, 96]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # [64*batch, 49, 96]
        # 2.2attention
        attention_windows = self.window_attention(x_windows)  # [64*batch, 49, 96]
        # 2.3merge windows
        attention_windows = attention_windows.view(-1, self.window_size, self.window_size, c)  # [64*batch, 7, 7, 96]
        # 2.4reverse window
        x = self.window_reverse(attention_windows, self.window_size, h, w)  # [batch, 56, 56, 96]
        x = x.view(n, h * w, c)  # [batch, 3136, 96]
        # 3 shortcut 1
        x_ = shortcut + self.drop_path(x)  # [batch, 3136, 96]
        # 4 layer norm
        x = self.layer_norm_2(x_)
        # 5 MLP
        x = self.mlp_1(x)
        # 6 shortcut 2
        x = x_ + self.drop_path(x)  # [batch, 3136, 96]
        # 7 layer norm
        x = self.layer_norm_3(x)
        x_ = x
        # 8 SW-MSA
        x = x.view(n, h, w, c)
        # 8.1cyclic shift windows
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        # 8.2partition windows
        shifted_x_windows = self.window_partition(shifted_x, self.window_size)  # [64*batch, 7, 7, 96]
        shifted_x_windows = shifted_x_windows.view(-1, self.window_size * self.window_size, c)  # [64*batch, 49, 96]
        # 8.3attention
        s_attention_windows = self.shift_window_attention(shifted_x_windows, self.attn_mask)  # [64*batch, 49, 96]
        # 8.4merge windows
        s_attention_windows = s_attention_windows.view(-1, self.window_size, self.window_size,
                                                       c)  # [64*batch, 7, 7, 96]
        # 8.5reverse window
        shifted_x = self.window_reverse(s_attention_windows, self.window_size, h, w)  # [batch, 56, 56, 96]
        # 8.6reverse cyclic shift windows
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(n, h * w, c)  # [batch, 3136, 96]
        # 9 shortcut 3
        x = x_ + self.drop_path(x)
        x_ = x
        # 10 layer norm
        x = self.layer_norm_3(x)
        # 11 mlp 2
        x = self.mlp_2(x)
        # 12 shortcut 4
        x = x_ + self.drop_path(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, in_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.liner = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.layer_norm = norm_layer(in_dim * 4)

    def forward(self, x):
        h, w = self.input_resolution
        n, sql_len, c = x.shape  # [batch, 3136, 96]
        assert sql_len == h * w, "input feature has wrong size"
        assert h % 2 == 0 and w % 2 == 0, f"x size ({h}*{w}) are not even."

        x = x.view(n, h, w, c)  # [batch, 56, 56, 96]

        x0 = x[:, 0::2, 0::2, :]  # [batch, 28, 28, 96]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [batch, 28, 28, 384]

        x = x.view(n, -1, 4 * c)
        x = self.layer_norm(x)
        x = self.liner(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, embed_dim, input_resolution, depth, num_head, window_size, norm_layer, is_last_layer=False):
        super().__init__()
        self.is_last_layer = is_last_layer
        # swin Transformer block 56->28->14->7
        self.stbs = nn.ModuleList([SwinTransformerBlock(in_dim=embed_dim, input_resolution=input_resolution,
                                                        num_heads=num_head, window_size=window_size,
                                                        shift_size=window_size // 2) for _ in range(depth)])

        # patch merging
        self.pm = PatchMerging(input_resolution=input_resolution, in_dim=embed_dim, norm_layer=norm_layer)

    def forward(self, x):
        for stb in self.stbs:
            x = stb(x)
        if not self.is_last_layer:
            x = self.pm(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, img_size=(224, 224, 3), patch_size=4, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # patch partition
        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, embedding_dim=embed_dim)
        patches_resolution = self.patch_embedding.patches_resolution

        # swin transformer layers
        self.swin_layers = nn.ModuleList([
            BasicLayer(embed_dim=int(embed_dim * 2 ** i),
                       input_resolution=(patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i)),
                       depth=depths[i],
                       num_head=num_heads[i], window_size=window_size, norm_layer=norm_layer,
                       is_last_layer=False if (i < self.num_layers - 1) else True)
            for i in range(self.num_layers)
        ])
        # print(self.swin_layers)
        # layer norm
        self.layer_norm = nn.LayerNorm(self.num_features)
        # average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # linear
        self.cls_head = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        # patch embedding
        x = self.patch_embedding(x)  # [batch, c, h, w] -> [batch, 3136, 96]
        # swin transformer layers
        for layer in self.swin_layers:
            x = layer(x)  # [batch, 3136, 96] -> [batch, 784, 192] -> [batch, 196, 384] -> [batch, 49, 768]
            print(x.shape)
        x = self.layer_norm(x)  # [batch, 49, 768]
        x = x.transpose(1, 2)  # [batch, 768, 49]
        # average pooling
        x = self.avg_pool(x)  # [batch, 768, 1]
        x = torch.flatten(x, 1)  # [batch, 768]
        # classify
        x = self.cls_head(x)  # [batch, class_num]
        return x


if __name__ == '__main__':
    images = torch.ones((2, 3, 224, 224))
    # images = torch.ones((2, 56*56, 96))
    # print(images)
    # pe = PatchEmbedding()
    # out = pe(images)
    st = SwinTransformer(img_size=(224, 224, 3),
                         patch_size=4,
                         num_classes=2,
                         embed_dim=96,
                         depths=[2, 2, 6, 2],
                         num_heads=[3, 6, 12, 24],
                         window_size=7)
    #
    # st = BasicLayer(embed_dim=96,
    #                 input_resolution=(56, 56),
    #                 num_heads=3,
    #                 window_size=7,
    #                 norm_layer=nn.LayerNorm)
    x = st(images)
    print(x.shape)
