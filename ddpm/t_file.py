# -*- coding: utf-8 -*-
"""
@File:        t_file.py
@Author:      Gene
@Software:    PyCharm
@Time:        05月 31, 2024
@Description: test code file
"""
import torch
import torch.nn as nn


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding


n_steps = 200
time_embedding_dim = 100

time_embed = nn.Embedding(n_steps, time_embedding_dim)
time_embed.weight.data = sinusoidal_embedding(n_steps, time_embedding_dim)
time_embed.requires_grad_(False)

time = torch.arange(0, n_steps, dtype=torch.long).reshape((n_steps, 1))  # pos：[max_len,1],即[1000,1]

print("time.shape: ", time.shape)
t = torch.randint(0, n_steps, (2,))

xx = time_embed(t)
print("t.shape: ", t.shape)
print("xx.shape: ", xx.shape)

a = pow(10000.0, torch.arange(0, time_embedding_dim, 2).float() / time_embedding_dim)

print(a.shape)
print(a)

b = torch.tensor([1 / 10000 ** (j / time_embedding_dim) for j in range(0, time_embedding_dim, 2)])

print(b.shape)
print(b)

c = time / a

d = time * b

print(c)

print(d)

print("*" * 100)

# Randomly generate tensor t with shape (2,)
t = torch.randint(0, 20, (2,))
print(t)
print(t.shape)

t = t.unsqueeze(1)
print(t)
print(t.shape)
# Expand the tensor to size [2, 200]
t = t.expand(2, 200)
print(t)
print(t.shape)

# Alternatively, you can use repeat to achieve the same result
repeated_tensor = t.repeat(1, 200)
print(repeated_tensor)
print(repeated_tensor.shape)
# # Initialize tensor x with shape (200, 100)
# x = torch.ones(20, 100)
#
# # Expand t to shape (2, 1) and then perform multiplication
# result = x[t]
# print(result)
# result = result.squeeze()
#
# print(result.shape)  # Should print torch.Size([2, 100])


import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, n_steps: int = 1000, time_embedding_dim: int = 100):
        super(TimeEmbedding, self).__init__()
        self.embedding = nn.Embedding(n_steps, time_embedding_dim)
        self.n_steps = n_steps
        # Initialize time_embedding as a buffer
        self.time_embedding = torch.zeros(n_steps, time_embedding_dim)
        time = torch.arange(0, n_steps, dtype=torch.float).reshape((n_steps, 1))  # time：[n_steps,1]
        div_term = time / pow(10000.0, torch.arange(0, time_embedding_dim, 2).float() / time_embedding_dim)
        self.time_embedding[:, 0::2] = torch.sin(div_term)
        self.time_embedding[:, 1::2] = torch.cos(div_term)
        # self.embedding.weight.data = time_embedding
        # 禁用梯度更新
        self.embedding.requires_grad_(False)

    def forward(self, t):
        t = self.embedding(t)  # [20, 100]
        t = t * self.time_embedding
        return t  # Select the corresponding embeddings


# Define the full model including time_mlp
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.time_mlp = TimeEmbedding(100, 100)

    def forward(self, t):
        return self.time_mlp(t)


# Usage example
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
t = torch.randint(0, 100, (20,)).to(device)
print(t.shape)
output = model(t)
print(output.shape)  # Expected shape: (2, 100)
