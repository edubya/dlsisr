
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRB(nn.Module):
    def __init__(self, N, k): # N = channels, k = kernel size
        super(CRB, self).__init__()
        self.conv1 = nn.Conv2d(N, N, k, stride=(1,1), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(N)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(N, N, k, stride=(1,1), padding=(1,1))
        self.prelu2 = nn.PReLU()
        self.bn2 = nn.BatchNorm2d(N)

        self.prelu3 = nn.PReLU()
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.prelu1(y)
        y = self.conv2(y)
        y = self.bn2(y)

        return self.prelu2(x + y)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__())