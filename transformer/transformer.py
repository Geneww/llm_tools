# -*- coding: utf-8 -*-
"""
@File:        transformer.py
@Author:      Gene
@Software:    PyCharm
@Time:        04月 20, 2023
@Description:
"""
import copy
import torch
import torch.nn as nn

import numpy as np


class PositionEncoder(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_token: int = 1000):
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 开始位置编码部分,先生成一个max_token * d_model 的矩阵，即1000 * 512
        # 1000是一个句子中最多的token数，512是一个token用多长的向量来表示，1000*512这个矩阵用于表示一个句子的信息
        self.pe = torch.zeros(1, max_token, d_model)
        pos = torch.arange(0, max_token, dtype=torch.float).unsqueeze(1)  # pos：[max_len,1],即[5000,1]
        # 先把括号内的分式求出来,pos是[5000,1],分母是[256],通过广播机制相乘后是[5000,256]
        div_term = pos / pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        # 再取正余弦
        self.pe[:, :, 0::2] = torch.sin(div_term)
        self.pe[:, :, 1::2] = torch.cos(div_term)
        # 将pe作为固定参数保存到缓冲区，不会被更新
        self.register_buffer('PositionEncoding', self.pe)

    def forward(self, x):
        '''x: [batch_size, seq_len, d_model]'''
        # 5000是我们预定义的最大的seq_len，就是说我们把最多的情况pe都算好了，用的时候用多少就取多少
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)  # return: [batch_size, seq_len, d_model], 和输入的形状相同


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
    def __init__(self, num_hidden=512, n_head=8, dropout=0.1, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_hidden = num_hidden
        self.num_head = n_head
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
        out = self.attention(q, k, v, mask)
        out = out.view(out.shape[0], -1, self.num_hidden)
        out = self.concat_mat(out)
        return out  # output: [batch_size, len_q, hidden_size]


class AddNorm(nn.Module):
    """
    使用残差和归一化
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.1, layer_norm_eps: float = 1e-5):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x, x_attention_out):
        return self.layer_norm(self.dropout(x_attention_out) + x)


class FeedForward(nn.Module):
    """
    前馈网络mlp
    """

    def __init__(self, d_model, dim_feedforward, activation):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 n_head: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation=nn.ReLU,
                 layer_norm_eps: float = 1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head, dropout)
        self.add_norm_1 = AddNorm(d_model, dropout, layer_norm_eps)
        self.feed_forward = FeedForward(d_model, dim_feedforward, activation)
        self.add_norm_2 = AddNorm(d_model, dropout, layer_norm_eps)

    def forward(self, x):
        x_attention_out = self.attention(x, x, x)
        x_norm_out = self.add_norm_1(x, x_attention_out)
        feed_out = self.feed_forward(x_norm_out)
        x = self.add_norm_2(x_norm_out, feed_out)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x):
        for net in self.layers:
            x = net(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 n_head: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation=nn.ReLU,
                 layer_norm_eps: float = 1e-5):
        super(TransformerDecoderLayer, self).__init__()
        self.mask_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.add_norm_1 = AddNorm(d_model, dropout, layer_norm_eps)
        self.attention = MultiHeadAttention(d_model, n_head, dropout)
        self.add_norm_2 = AddNorm(d_model, dropout, layer_norm_eps)
        self.feed_forward = FeedForward(d_model, dim_feedforward, activation)
        self.add_norm_3 = AddNorm(d_model, dropout, layer_norm_eps)

    def forward(self, x, encoder_out):
        out = self.mask_attention(x, x, x, True)
        print("mask_attention_out: ", out.shape)
        norm_out = self.add_norm_1(x, out)
        print("norm_out_1: ", norm_out.shape)
        out = self.attention(encoder_out, encoder_out, norm_out)
        print("attention_out: ", out.shape)
        norm_out = self.add_norm_2(norm_out, out)
        print("norm_out_2: ", norm_out.shape)
        out = self.feed_forward(norm_out)
        print("feed_out: ", out.shape)
        norm_out = self.add_norm_3(norm_out, out)
        print("norm_out_3: ", norm_out.shape)
        return norm_out


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, x, encoder_out):
        for net in self.layers:
            x = net(x, encoder_out)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int = 1000, d_model: int = 512, n_head: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation=nn.ReLU,
                 layer_norm_eps: float = 1e-5):
        super(Transformer, self).__init__()
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.output_embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码 解决词向量之间位置顺序问题
        self.in_position_encoder = PositionEncoder(d_model=d_model, dropout=dropout)
        self.out_position_encoder = PositionEncoder(d_model=d_model, dropout=dropout)
        # encoder
        encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, activation, layer_norm_eps)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        # decoder
        decoder_layer = TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, activation, layer_norm_eps)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        # linear
        self.linear = nn.Linear(d_model, d_model)
        # softmax
        self.softmax = nn.Softmax(-1)

    def forward(self, x, y):
        x = self.input_embedding(x)
        print(x.shape)
        x = self.in_position_encoder(x)

        y = self.output_embedding(y)
        y = self.out_position_encoder(y)
        # encoder
        x = self.encoder(x)
        # decoder
        x = self.decoder(x, y)
        # out
        x = self.linear(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    transformer = Transformer()
    src = torch.LongTensor([[1, 1, 1],
                            [1, 1, 1]])
    tgt = torch.LongTensor([[1, 1, 1],
                            [1, 1, 1]])
    out = transformer(src, tgt)
