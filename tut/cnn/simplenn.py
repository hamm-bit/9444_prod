import pandas as pd
import numpy as np
import torch
import random

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from os.path import join
from mnistshow import MnistDataloader

# ================================================================
# Computation config
# ================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epoch = int(1e5)
batch_size = 100
alpha = 0.02

# ================================================================
# Collects data and pre-process for training
# ================================================================

input_path = '../archive/'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(img_train, y_train), (img_test, y_test) = mnist_dataloader.load_data()

x_train = img_train.reshape(img_train.shape[0], -1)
x_test = img_test.reshape(img_test.shape[0], -1)

# Move training data to CUDA
x_train_tensor = x_train.astype(np.float32) / 255.0
x_train_tensor = x_train_tensor.reshape(-1, 1, 28, 28)      # (batch_size, 1, 28, 28)
x_train_tensor = torch.tensor(x_train_tensor).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

# Move testing data to CUDA
x_test_tensor = x_test.astype(np.float32) / 255.0
x_test_tensor = x_test_tensor.reshape(-1, 1, 28, 28)      # (batch_size, 1, 28, 28)
x_test_tensor = torch.tensor(x_train_tensor).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# ================================================================
# ================================================================
# ================================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=0),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.l2 = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=5, padding=0),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU())
        self.fc1 = nn.Linear(16*16, 8*8)            # DENSE
        self.fc2 = nn.Linear(8*8, 10)               # DENSE to output
    
    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = out.view(-1, 256)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# After initialization, we set the network
mnistnet = SimpleCNN().to(device)

# Give it the loss function (criterion)
# And the optimizer
crit = nn.CrossEntropyLoss()         # defaults to reduction='mean'
optimizer = opt.SGD(mnistnet.parameters(), lr=0.1)

# Sample dset
# Import MNIST set
index = 1
tot_size = len(x_train)
for epoch in range(num_epoch):
    for (image, label) in zip(x_train, y_train):
        inputs = x_train_tensor[index : index + batch_size]
        labels = y_train_tensor[index : index + batch_size]

        # Forward
        outputs = mnistnet(inputs)
        # print(f'output length = {len(outputs)}, input length = {len(labels)}')
        loss = crit(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            print(f'Epoch [{epoch} / {num_epoch}], step [{index} / {tot_size}], loss: {loss.item():.4f}')
print("Training complete")

# Tester
test_size = len(x_test)
mnistnet.eval()
with torch.no_grad():
    correct = 0; total = 0
    for (image, label) in zip(x_test, y_test):
        image = image.to(device)
        output = mnistnet(image)
        _, predicted = torch.max(output.data, 1)
        total += 1
        correct += (predicted == label).sum().item()
        print(f'Test accuracy of model on the {x_test} images: {100 * correct / total:.2f}%')

torch.save(mnistnet.state_dict(), 'model.ckpt')
