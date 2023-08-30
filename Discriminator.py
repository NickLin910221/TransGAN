import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(28 * 28, 392),
            nn.ReLU(),
            nn.Linear(392, 98),
            nn.ReLU(),
            nn.Linear(98, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.disc(x)
        return x