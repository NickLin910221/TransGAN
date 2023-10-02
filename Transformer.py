import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, attention_heads = 1, layer = 3, size = (28, 28)) -> None:
        self.attention_heads = attention_heads

        self.query = torch.randn(layer, attention_heads, size[0], size[1])
        self.key = torch.randn(layer, attention_heads, size[0], size[1])
        self.value = torch.randn(layer, attention_heads, size[0], size[1])

    def forward(self):

    def Attention(self):
        return