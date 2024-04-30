# -*- coding: utf-8 -*-
"""
@File:        my_transformer.py
@Author:      Gene
@Software:    PyCharm
@Time:        04月 18, 2024
@Description:
"""
import math

import torch
from torch import nn
import numpy as np


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

    def __init__(self, ffn_hidden_size, ffn_num_output):
        super(FeedForward, self).__init__()
        self.dense1 = nn.LazyLinear(ffn_hidden_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_hidden_size, ffn_num_output)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))


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
        print("q.shape: ", q.shape)
        print("v.shape: ", v.shape)
        out = self.attention(q, k, v, mask)
        print("attention shape: ", out.shape)
        out = out.view(out.shape[0], -1, self.num_hidden)
        out = self.concat_mat(out)
        # print(out.shape)
        return out  # output: [batch_size, len_q, hidden_size]


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size=512, head_size=8):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(hidden_size, head_size)
        self.add_norm = AddNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, hidden_size)

    def forward(self, x):
        x_attention_out = self.attention(x, x, x)
        print("x_attention_out: ", x_attention_out.shape)
        x_norm_out = self.add_norm(x, x_attention_out)
        print("x_norm_out: ", x_norm_out.shape)
        feed_out = self.feed_forward(x_norm_out)
        x = self.add_norm(x_norm_out, feed_out)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size=10000, hidden_size=768, attention_heads_size=12, n_layers=6):
        super(Encoder, self).__init__()
        self.in_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(vocab_size, hidden_size)
        self.encoder_layers = nn.Sequential(
            *[EncoderLayer(hidden_size, attention_heads_size) for _ in range(n_layers)])

    def forward(self, x):
        x = self.in_embedding(x)
        x = self.pos_encoding(x)
        print("1: ", x.shape)
        # print(self.encoder_layers)
        for layer in self.encoder_layers:
            x = layer(x)
            print(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size=768, attention_heads_size=12):
        super(DecoderLayer, self).__init__()
        self.masked_attention = MultiHeadAttention(hidden_size, attention_heads_size)
        self.add_norm_1 = AddNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, attention_heads_size)
        self.add_norm_2 = AddNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size * 4, hidden_size)
        self.add_norm_3 = AddNorm(hidden_size)

    def forward(self, x, encoder_input):
        # encoder attention
        x_masked_attention_out = self.masked_attention(x, x, x, True)
        x_norm_out = self.add_norm_1(x, x_masked_attention_out)
        print("encoder_input.shape: ", encoder_input.shape)
        print("x_norm_out.shape: ", x_norm_out.shape)
        # encode decode attention
        x_attention_out = self.attention(x_norm_out, encoder_input, encoder_input)
        x_norm_out = self.add_norm_2(x_norm_out, x_attention_out)
        x = self.feed_forward(x_norm_out)
        x = self.add_norm_3(x, x_norm_out)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, attention_heads_size, n_layers=6):
        super(Decoder, self).__init__()
        self.out_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(vocab_size, hidden_size)
        self.decoder_layers = nn.Sequential(*[DecoderLayer(hidden_size, attention_heads_size) for _ in range(n_layers)])

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_input, encoder_input):
        x = self.out_embedding(decoder_input)
        x = self.pos_encoding(x)
        print("3: ", x.shape)

        for layer in self.decoder_layers:
            print("2: ", encoder_input.shape)
            x = layer(x, encoder_input)
        x = self.linear(x)
        print("*"*100)
        print(x.shape)
        x = self.softmax(x)
        return x


class Transformer(nn.Module):
    def __init__(self, e_vocab_size=1000, d_vocab_size=1000, hidden_size=512, n_head=12, dropout=0.1):
        super(Transformer, self).__init__()
        # 编码器
        self.encoder_embedding = nn.Embedding(e_vocab_size, hidden_size)
        self.encoder_pos_encoding = PositionalEncoding(e_vocab_size, hidden_size)
        self.encoder = Encoder(e_vocab_size, hidden_size, n_head, 6)
        # 解码器
        self.decoder_out_embedding = nn.Embedding(d_vocab_size, hidden_size)
        self.decoder_pos_encoding = PositionalEncoding(d_vocab_size, hidden_size)
        self.decoder = Decoder(d_vocab_size, hidden_size, n_head, 6)
        # out
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(-1)

    def forward(self, x, y):

        x_encoder_out = self.encoder(x)

        # if self.training:
        #     x_encoder_out = y
        print("*"*88)
        print("y: ", y.shape)
        print("x_encoder_out: ", x_encoder_out.shape)
        x = self.decoder(y, x_encoder_out)
        x = self.linear(x)
        x = self.softmax(x)
        return x


sentence = [
    # enc_input   dec_input    dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E'],
]

# 词典，padding用0来表示
# 源词典
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocab_size = len(src_vocab)  # 6
# 目标词典（包含特殊符）
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
tgt_vocab_size = len(tgt_vocab)  # 9


def make_data(sentence):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentence)):
        enc_input = [src_vocab[word] for word in sentence[i][0].split()]
        dec_input = [tgt_vocab[word] for word in sentence[i][1].split()]
        dec_output = [tgt_vocab[word] for word in sentence[i][2].split()]

        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    # LongTensor是专用于存储整型的，Tensor则可以存浮点、整数、bool等多种类型
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


import torch.utils.data as Data


# 使用Dataset加载数据
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        # 我们前面的enc_inputs.shape = [2,5],所以这个返回的是2
        return self.enc_inputs.shape[0]

        # 根据idx返回的是一组 enc_input, dec_input, dec_output

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]



def bleu(pred_seq, label_seq, k):
    """计算BLEU

    Defined in :numref:`sec_seq2seq_training`"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

enc_inputs, dec_inputs, dec_outputs = make_data(sentence)

# # 构建DataLoader
loader = Data.DataLoader(dataset=MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=2, shuffle=True)

transfer = Transformer(src_vocab_size, tgt_vocab_size, 12)

for enc_inputs, dec_inputs, dec_outputs in loader:
    print("enc_inputs.shape: ", enc_inputs.shape)
    print("dec_inputs.shape: ", dec_inputs.shape)
    out = transfer(enc_inputs, dec_inputs)
    print(out, out.shape)

quit()
