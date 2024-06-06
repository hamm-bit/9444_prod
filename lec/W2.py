import torch
import matplotlib as plt

# example program in class
'''
net = myNetwork().to(GPU)

train_loader = torch.utils.data.DataLoader(...)
test_loader = torch.utils.data.DataLoader(...)

optimizer = torch.optim.SGD(net.params, ...)

for epoch in range(1, epochs): # training loop
    train(args, net, device, train loader, optimizer)
    # periodically evaluate network on test data
    if epoch % 10 == 0:
        test( args, net, device, test loader)

def __init__(self):
    super(myNetwork)
    # define structure of the network here
def forward(self, input):
    # define inference rules
    pass

'''

# Live example being
import torch.nn as nn
class MyModel(nn.Module):
    def init (self):
        super(MyModel, self).__init__()
        self.A = nn.Parameter(torch.randn((1),requires_grad=True))
        self.B = nn.Parameter(torch.randn((1),requires_grad=True))

    def forward(self, input):
        output = self.A * input[:,0] * torch.log(input[:,1]) \
        + self.B * input[:,1] * input[:,1]
        return output

class MyModelDef(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.in_to_hid = torch.nn.Linear(2,2)               # layer with 2 inputs and 2 outputs 
        self.hid_to_out = torch.nn.Linear(2,1)              # layer with 2 inputs and 1 output
    def forward(self, input):
        hid_sum = self.in_to_hid(input)
        hidden = torch.tanh(hid_sum)
        out_sum = self.hid_to_out(hidden)
        output = torch.sigmoid(out_sum)
        return output

'''
Network layers:
    nn.linear
    nn.Conv2d
Intermediate operator
    nn.Dropout
    nn.BatchNorm
Activation functions
    nn.Sigmoid
    nn.Tanh
    nn.ReLU
'''

'''
from data import ImageFolder
    dataset = ImageFolder(folder, transform)
'''

import torchvision.datasets as dsets
    mnistset = dsets.MNIST(...)
    cifarset = dsets.CIFAR10(...)
    celebset = dsets.CelebA(...)
