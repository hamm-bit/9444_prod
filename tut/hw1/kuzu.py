"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)
        # INSERT CODE HERE

    def forward(self, x):
        out = x.view(-1, 28 * 28)
        out = self.fc1(out)
        return F.log_softmax(out)

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = x.view(-1, 28 * 28)
        out = F.tanh(self.fc1(out))
        return F.log_softmax(self.fc2(out))

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=0),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(16, 256, 5, padding=0),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(256 * 4 * 4, 10)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        # print(f"shape of data: {out.shape}")
        out = out.view(-1, 4096)
        return F.log_softmax(self.fc1(out))

