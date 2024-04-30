# -*- coding: utf-8 -*-
"""
@File:        t1.py
@Author:      Gene
@Software:    PyCharm
@Time:        04月 18, 2024
@Description:
"""
import torch
from torch import nn
import numpy as np

print(-1e9)
a = torch.randn((2, 4))
print(a)
b = torch.randn((2, 4))
print(b)
print(b.T)
c = torch.matmul(a, b.T)

print(c)

print(a @ b.T)

d_k = 64

aa = np.sqrt(d_k)

print(aa)

bb = a @ b.T / np.sqrt(d_k)
print(bb)

cc = bb * 8
print(cc)

print(512 // 8)

print("*"*64)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # seq_k.data.eq(0)返回一个等大的布尔张量，seq_k元素等于0的位置为True,否则为False
    # 然后扩维以保证后续操作的兼容(广播)
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # pad_attn_mask: [batch_size,1,len_k]
    # 要为每一个q提供一份k，所以把第二维度扩展了q次
    # 另注意expand并非真正加倍了内存，只是重复了引用，对任意引用的修改都会修改原始值
    # 这里是因为我们不会修改这个mask所以用它来节省内存
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # return: [batch_size, len_q, len_k]
    # 返回的是batch_size个 len_q * len_k的矩阵，内容是True和False，
    # 第i行第j列表示的是query的第i个词对key的第j个词的注意力是否无意义，若无意义则为True，有意义的为False（即被padding的位置是True）


# 用于获取对后续位置的掩码，防止在预测过程中看到未来时刻的输入
# 原文：to prevent positions from attending to subsequent positions
def get_attn_subsequence_mask(seq):
    """seq: [batch_size, tgt_len]"""
    # batch_size个 tgt_len * tgt_len的mask矩阵
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # np.triu 是生成一个 upper triangular matrix 上三角矩阵，k是相对于主对角线的偏移量
    # k=1意为不包含主对角线（从主对角线向上偏移1开始）
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # 因为只有0、1所以用byte节省内存
    return subsequence_mask  # return: [batch_size, tgt_len, tgt_len]

seq_q = torch.randn((1, 10))

mask = get_attn_pad_mask(seq_q, seq_q)
# print(mask)

dec_self_attn_subsequence_mask = get_attn_subsequence_mask(seq_q)
# print(dec_self_attn_subsequence_mask)

# 将两个mask叠加，布尔值可以视为0和1，和大于0的位置是需要被mask掉的，赋为True，和为0的位置是有意义的为False
dec_self_attn_mask = torch.gt((mask + dec_self_attn_subsequence_mask), 0)

# print(dec_self_attn_mask)
# 这是co-attention部分，为啥传入的是enc_inputs而不是enc_outputs呢


transformer = nn.Transformer(2, 2, 6, 6, 2048, 0.1)

src = torch.rand((2, 2, 2))
print(src)
tgt = torch.rand((2, 2, 2))
print(tgt)
out = transformer(src, tgt)

def generate_mask(dim):
    # 此处是 sequence mask ，防止 decoder窥视后面时间步的信息。
    # padding mask 在数据输入模型之前完成。
    matirx = np.ones((dim, dim))
    mask = torch.Tensor(np.tril(matirx))
    return mask == 0

mask = generate_mask(10)
# print(mask)

attention_score = np.ones((10, 10))

attention_score.masked_fill(mask, value=float("-inf"))  # 注意这里的小Trick，不需要将Q,K,V 分别MASK,只MASKSoftmax之前的结果就好了


