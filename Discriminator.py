import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, batch_size) -> None:
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input 1 * 28 * 28
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 16),
            # -> 1 * 13 * 13
            nn.ReLU(),
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 8),
            # -> 1 * 6 * 6
            nn.ReLU(),
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 4),
            # -> 1 * 3 * 3
            nn.ReLU(),
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.disc(x)
        return x