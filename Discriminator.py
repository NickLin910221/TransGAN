import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, batch_size) -> None:
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input 1 * 28 * 28
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 5),
            nn.MaxPool2d(2, 2),
            # -> 1 * 12 * 12, 
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 5),
            nn.MaxPool2d(2, 2),
            # -> 1 * 4 * 4, 
            nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3),
            nn.MaxPool2d(2, 2),
            # -> 1 * 1 * 1, 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.disc(x)
        return x