import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, layer = 3, attention_heads = 1, size = (28, 28), dropout = 0.1) -> None:
        super(Transformer, self).__init__()
        self.attention_heads = attention_heads
        self.layer = layer

        self.query = torch.randn(layer, attention_heads, size[0], size[1])
        self.key = torch.randn(layer, attention_heads, size[0], size[1])
        self.value = torch.randn(layer, attention_heads, size[0], size[1])
        self.dropout = [nn.Dropout(dropout) for i in range(layer)]

    def forward(self, x):
        for l in range(self.layer):
            for h in range(self.attention_heads):
                x += torch.matmul(x, self.attention(self.query[l][h], self.key[l][h], self.value[l][h]))  
            x += self.dropout[l](x)
        return x

    def attention(self, q, k, v):
        score = torch.matmul(torch.matmul(q, k.permute(0, 1)), v)
        score = F.softmax(score, dim=-1)
        return score