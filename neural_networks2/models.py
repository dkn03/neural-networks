import torch.nn.functional as F
from torch import nn
from torch import tanh


class CustomModel(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        out = F.max_pool2d(tanh(self.conv1(x)), 2)
        out = F.max_pool2d(tanh(self.conv2(out)), 2)
        out = out.view(-1, 32 * 32 * self.n_chans1 // 2)
        out = tanh(self.fc1(out))
        out = F.softmax(self.fc2(out))
        return out