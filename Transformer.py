import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, device, layer = 3, attention_heads = 1, size = (28, 28), dropout = 0.1) -> None:
        super(Transformer, self).__init__()
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

        self.dropout = [nn.Dropout(dropout) for l in range(layer)]

    def forward(self, x):
        num = x.shape[0]
        channel = x.shape[1]

        for l in range(self.layer):
            heads = []
            for r in range(self.attention_heads):
                for c in range(self.attention_heads):
                    a = torch.matmul(x[:,:,self.r * r:self.r * (r + 1),self.c * c:self.c * (c + 1)], self.attention(self.query[l][r][c], self.key[l][r][c], self.value[l][r][c]))
                    heads.append(a)
            # heads = torch.cat(heads, 1)
            out = torch.Tensor(num, channel, self.attention_heads ** 2, self.r, self.c)
            torch.cat(heads, out = out)

            # heads = heads.view(num, channel)
            print(heads.shape)
            # x += self.dropout[l](heads)
        # return x

    def attention(self, q, k, v):
        score = torch.matmul(torch.matmul(q, k.permute(0, 1)), v)
        score = F.softmax(score, dim=-1)
        return score