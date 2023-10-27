import torch.nn as nn
import torch.nn.functional as F
from Transformer import Transformer

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.trans = Transformer(attention_heads = 4)
        self.gen = nn.Sequential(
            # Input 1 * 28 * 28
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, padding = 1),
            # -> 1 * 28 * 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x