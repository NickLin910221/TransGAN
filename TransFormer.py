import torch.nn as nn
import torch.nn.functional as F

class TransFormer(nn.Module):
    def __init__(self) -> None:
        super(TransFormer, self).__init__()
        self.nn1 = nn.Linear(1, 1024)
        self.nn2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.relu(self.nn1(x))
        x = self.nn2(x)
        return x