import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, batch_size) -> None:
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 15)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7 * 7, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 1 * 7 * 7)
        x = F.sigmoid(self.fc1(x))
        return x