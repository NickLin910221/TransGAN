import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, device, layer = 3, attention_heads = 1, size = (28, 28), dropout = 0.1) -> None:
        super(Transformer, self).__init__()
        self.attention_heads = attention_heads
        self.layer = layer

        self.device = device

        assert (size[0] % attention_heads == 0 and size[1] % attention_heads == 0), "attention_heads can't slice the image."

        self.query = nn.Parameter(torch.randn(layer, size[0] / attention_heads, size[1] / attention_heads, attention_heads, attention_heads))
        self.key = nn.Parameter(torch.randn(layer, size[0] / attention_heads, size[1] / attention_heads, attention_heads, attention_heads))
        self.value = nn.Parameter(torch.randn(layer, size[0] / attention_heads, size[1] / attention_heads, attention_heads, attention_heads))

        self.dropout = [nn.Dropout(dropout) for l in range(layer)]

    def forward(self, x):
        for l in range(self.layer):
            for r in range(self.attention_heads):
                for c in range(self.attention_heads):
                    heads = []
                    heads.append(torch.matmul(x, self.attention(self.query[l][r][c], self.key[l][r][c], self.value[l][r][c])))
                    heads = torch.cat(heads, 1)
            x += self.dropout[l](heads)
        return x

    def attention(self, q, k, v):
        score = torch.matmul(torch.matmul(q, k.permute(0, 1)), v)
        score = F.softmax(score, dim=-1)
        return score