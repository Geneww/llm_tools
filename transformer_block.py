# -*- coding: utf-8 -*-
"""
@File:        transformer_block.py
@Author:      Gene
@Software:    PyCharm
@Time:        04月 13, 2024
@Description:
"""
import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, sql_len, hidden_size, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = torch.zeros((1, sql_len, hidden_size))

        pos = torch.arange(0, sql_len, dtype=torch.float32).reshape(-1, 1)
        div_term = pos / torch.pow(10000.0, torch.arange(0, hidden_size, 2, dtype=torch.float32) / hidden_size)

        self.pe[:, :, 0::2] = torch.sin(div_term)
        self.pe[:, :, 1::2] = torch.cos(div_term)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def get_attention_mask(seq):
        """seq: [batch_size, tgt_len]"""
        # batch_size个 tgt_len * tgt_len的mask矩阵
        # attn_shape = [seq.size(0), seq.size(1), seq.size(2)]
        attn_shape = seq.shape
        print("attn_shape: ", attn_shape)
        # np.triu 是生成一个 upper triangular matrix 上三角矩阵，k是相对于主对角线的偏移量
        # k=1意为不包含主对角线（从主对角线向上偏移1开始）
        subsequence_mask = np.triu(np.ones(attn_shape), k=1)
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # 因为只有0、1所以用byte节省内存
        print("subsequence_mask: ", subsequence_mask.shape)
        return subsequence_mask == 1  # return: [batch_size, n_head, tgt_len, tgt_len]

    def forward(self, q, k, v, mask=False):
        # scores = q @ k / np.sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(q.shape[-1])  # [batch, n_head, len_q, len_k]
        if mask:
            # 生成mask掩码
            attention_mask = self.get_attention_mask(scores)
            scores.masked_fill_(attention_mask, -1e9)
        return torch.matmul(self.dropout(self.softmax(scores)), v)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hidden=512, num_head=8, dropout=0.1, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_hidden = num_hidden
        self.num_head = num_head
        self.q_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.k_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.v_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.concat_mat = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, queries, keys, values, mask=False):
        # 分多个头  [batch, seq_len, hidden_size] -> [batch, seq_len, head_size, d_k] -> [batch, head_size, seq_len, d_k]
        d_k = self.num_hidden // self.num_head
        q = self.q_mat(queries).view(queries.size(0), -1, self.num_head, d_k).transpose(1, 2)
        k = self.k_mat(keys).view(keys.size(0), -1, self.num_head, d_k).transpose(1, 2)
        v = self.v_mat(values).view(values.size(0), -1, self.num_head, d_k).transpose(1, 2)
        print("v.shape: ", v.shape)
        out = self.attention(q, k, v, mask)
        print("attention shape: ", out.shape)
        out = out.view(out.shape[0], -1, self.num_hidden)
        out = self.concat_mat(out)
        # print(out.shape)
        return out  # output: [batch_size, len_q, hidden_size]


class AddNorm(nn.Module):
    """
    使用残差和归一化
    """

    def __init__(self, hidden_size, dropout=0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, x_attention_out):
        return self.layer_norm(self.dropout(x_attention_out) + x)


class FeedForward(nn.Module):
    """
    前馈网络mlp
    """

    def __init__(self, ffn_num_input, ffn_hidden_size, ffn_num_output):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_hidden_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_hidden_size, ffn_num_output)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size=512, head_size=8, ffn_num_input=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(num_hidden=hidden_size, num_head=head_size, dropout=dropout)
        self.add_norm = AddNorm(hidden_size)
        self.feed_forward = FeedForward(ffn_num_input, hidden_size, hidden_size)

    def forward(self, x):
        x_attention_out = self.attention(x, x, x)
        x_norm_out = self.add_norm(x, x_attention_out)
        feed_out = self.feed_forward(x_norm_out)
        x = self.add_norm(x_norm_out, feed_out)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size=10000, hidden_size=768, attention_heads_size=12, n_layers=6):
        super(Encoder, self).__init__()
        self.in_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(vocab_size, hidden_size)
        self.encoder_layers = nn.Sequential(
            *[EncoderLayer(hidden_size, attention_heads_size, hidden_size * 4) for _ in range(n_layers)])

    def forward(self, x):
        x = self.in_embedding(x)
        x = self.pos_encoding(x)
        # print(self.encoder_layers)
        for layer in self.encoder_layers:
            x = layer(x)
            print(x)
        return x


class MLP(nn.Module):
    def __int__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.act = act_layer()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(in_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


num_hiddens, num_heads = 10, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.1)
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
print(X.shape)
print(Y.shape)
a = attention(X, X, X, True)
e = Encoder(10, num_hiddens, num_heads, 6)
a = e(X)
print(a)