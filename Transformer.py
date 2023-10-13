import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, activation_func = nn.LeakyReLU, dropout = 0) -> None:
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
    def __init__(self, d_model, size = (4, 4), max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.size = size
        max_len = self.size[0] * self.size[1]
        self.pe = torch.zeros(max_len, d_model)
        self._pe_ = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        pe = self.pe.expand(-1, int(x.shape[2] / self.size[0]))
        pe = pe.reshape(-1, x.shape[2]).unsqueeze(1)
        print(pe)
        pe = pe.expand(-1, int(x.shape[3] / self.size[1]), -1)
        print(pe)
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                self._pe_[][]
                # print(x[:, :, int(x.shape[2] / self.size[0] * r):int(x.shape[2] / self.size[0] * (r + 1)), int(x.shape[3] / self.size[1] * c):int(x.shape[3] / self.size[1] * (c + 1))])
                # x[:, :, int(x.shape[2] / self.size[0] * r):int(x.shape[2] / self.size[0] * (r + 1)), int(x.shape[3] / self.size[1] * c):int(x.shape[3] / self.size[1] * (c + 1))] + self.pe[self.size[0] * r + c]
                # print(x[:, :, int(x.shape[2] / self.size[0] * r):int(x.shape[2] / self.size[0] * (r + 1)), int(x.shape[3] / self.size[1] * c):int(x.shape[3] / self.size[1] * (c + 1))])
        self.pe

class Transformer(nn.Module):

    def __init__(self, device, layer = 3, attention_heads = 1, size = (28, 28), dropout = 0.1) -> None:
        super(Transformer, self).__init__()
        self.PE = PositionalEncoding(1, size = (4, 4), max_len = attention_heads ** 2).to(device)

        self.attention_heads = attention_heads
        self.layer = layer
        self.size = size

        self.device = device

        assert (size[0] % attention_heads == 0 and size[1] % attention_heads == 0), "attention_heads can't slice the image."

        self.r = int(size[0] / attention_heads)
        self.c = int(size[1] / attention_heads)

        self.query = nn.Parameter(torch.randn(layer, attention_heads, attention_heads, self.r, self.c))
        self.key = nn.Parameter(torch.randn(layer, attention_heads, attention_heads, self.r, self.c))
        self.value = nn.Parameter(torch.randn(layer, attention_heads, attention_heads, self.r, self.c))

        self.MLP = [MLP(size[0] * size[1]).to(device) for l in range(layer)]
        self.norm = nn.LayerNorm([size[0], size[1]])
        self.Tanh = nn.Tanh()

    def forward(self, x):
        num = x.shape[0]
        channel = x.shape[1]
        self.PE(x)
        for l in range(self.layer):
            heads = []
            x1 = self.norm(x)
            for r in range(self.attention_heads):
                for c in range(self.attention_heads):
                    heads.append(torch.matmul(x1[:,:,self.r * r:self.r * (r + 1),self.c * c:self.c * (c + 1)], self.attention(self.query[l][r][c], self.key[l][r][c], self.value[l][r][c])))
            for r in range(self.attention_heads):
                for c in range(1, self.attention_heads):
                    heads[r * self.attention_heads] = torch.cat((heads[r * self.attention_heads], heads[r * self.attention_heads + c]), dim = 2)
                if r > 0:
                    heads[0] = torch.cat((heads[0], heads[r * self.attention_heads]), dim = 3)
            x2 = x + heads[0]
            x3 = self.norm(x2).view(num, channel, -1)
            # x4 = self.MLP[l](x3).view(num, channel, self.size[0], self.size[1])
            # x = x4 + x2
        x = self.Tanh(x)
        return x

    def attention(self, q, k, v):
        score = torch.matmul(q, k.permute(0, 1))
        score = F.softmax(score, dim = -1)
        score = torch.matmul(score, v)
        return score