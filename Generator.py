import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        x = x.view(-1, 28, 28)
        return x