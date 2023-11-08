import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from torchvision.utils import save_image

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, activation_func = nn.LeakyReLU, dropout = 0.1) -> None:
        super(MLP, self).__init__()

        out_features = in_features or out_features
        hidden_features = in_features or hidden_features
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.activation_func = activation_func()
        self.layer2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation_func(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, row = 28, col = 28):
        super(PositionalEncoding, self).__init__()

        max_len = row * col
        self.row = row
        self.col = col
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = nn.Parameter(torch.sin(position * div_term))
        pe[:, 1::2] = nn.Parameter(torch.cos(position * div_term))

        self.register_buffer('pe', pe.view(row, col))

    @torch.no_grad()
    def forward(self, x):
        return x + self.pe

class Transformer(nn.Module):

    def __init__(self, attention_heads = 1, size = (28, 28), dropout = 0.1) -> None:
        super(Transformer, self).__init__()
        self.PE = PositionalEncoding(1)

        self.attention_heads = attention_heads

        assert (size[0] % attention_heads == 0 and size[1] % attention_heads == 0), "attention_heads can't slice the image."

        self.r = int(size[0] / attention_heads)
        self.c = int(size[1] / attention_heads)

        self.qkv = nn.Linear(size[0] * size[1], size[0] * size[1] * 3)
        self.MLP = MLP(size[0] * size[1])

        self.softmax = nn.Softmax(dim = -1)
        self.norm = nn.LayerNorm([size[0], size[1]])
        self.Tanh = nn.Tanh()

    def forward(self, x):
        save_image(x[:64], f"./x.png")
        x_withpe = self.PE(x)

        qkv = self.qkv(x_withpe.view(x_withpe.shape[0], x_withpe.shape[1], -1)).view(x_withpe.shape[0], x_withpe.shape[1], self.attention_heads ** 2, 3, self.r, self.c)
        query, key, value = qkv.unbind(3)

        attention = self.attention(query, key, value, self.attention_heads ** 2)
        heads = []

        for r in range(self.attention_heads):
            for c in range(self.attention_heads):
                heads.append(attention[:, :, r * self.attention_heads + c])

        for r in range(self.attention_heads):
            for c in range(1, self.attention_heads):
                heads[r * self.attention_heads] = torch.cat((heads[r * self.attention_heads], heads[r * self.attention_heads + c]), dim = 2)
            if r > 0:
                heads[0] = torch.cat((heads[0], heads[r * self.attention_heads]), dim = 3)

        x = x + self.norm(heads[0])
        save_image(x[:64], f"./middle.png")
        x2 = x.view(x.shape[0], x.shape[1], -1)
        x3 = self.MLP(x2).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        save_image(x3[:64], f"./MLP.png")
        x = x + x3
        x = self.Tanh(x)
        save_image(x[:64], f"./output.png")
        return x

    def attention(self, q, k, v, d):
        score = torch.matmul(q, k.permute(0, 1, 2, 4, 3)) * (d ** -0.5)
        score = self.softmax(score)
        score = torch.matmul(score, v)
        return score