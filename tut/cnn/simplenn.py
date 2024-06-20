import pandas as pd
import numpy as np
import torch
import random
import sklearn
import os

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from sklearn.metrics import classification_report, log_loss
from os.path import join
from mnistshow import MnistDataloader

# ================================================================
# Network definition
# ================================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.l2 = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=5, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU())
        # Added 2nd dense layer: moar accuracy
        self.fc1 = nn.Linear(16*16, 96)            # DENSE 1
        self.fc2 = nn.Linear(96, 32)               # DENSE 2
        self.fc3 = nn.Linear(32, 10)               # DENSE to output
    
    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = out.view(-1, 256)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# ================================================================
# Computation config
# ================================================================
print(f"Cuda is detected on machine: {torch.cuda.is_available()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# After initialization, we set the network
mnistnet = SimpleCNN().to(device)

num_epoch = int(10)
batch_size = 100
alpha = 8e-5

print('===============================================')
print('======== Neural network for MNIST Dset ========')
print('===============================================')
detected = os.path.exists('mnist_model.ckpt')
print(f'Detected trained model: {detected}')
print(f'Type:\n\t0 to (re-)train the model, \n\t1 to inference')
if not ((int(input()) == 1) and detected):
    print("Retraining starts: ...")

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
    x_test_tensor = torch.tensor(x_test_tensor).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)



# Give it the loss function (criterion)
# And the optimizer
    crit = nn.CrossEntropyLoss()         # defaults to reduction='mean'
    optimizer = opt.Adam(mnistnet.parameters(), lr=alpha)

# Sample dset
# Import MNIST set
    index = 0
    tot_size = len(x_train)
    for epoch in range(num_epoch):
        tot_loss = 0
        for (image, label) in zip(x_train, y_train):
            index %= (tot_size - batch_size)
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
            tot_loss += loss.item()

            if index % 10000 == 0:
                print(f'Epoch [{epoch} / {num_epoch}], step [{index} / {tot_size}], loss: {tot_loss:.4f}')
            index += 1
        print(f'Final mean of loss: {(tot_loss / 60000):.4f}')
    print("Training complete")

    # Tester
    # Evaluation
    mnistnet.eval()
    y_true = []
    y_pred = []
    y_pred_proba = []

    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            images = x_test_tensor[i : i + batch_size].to(device)
            labels = y_test_tensor[i : i + batch_size].to(device)
            outputs: torch.Tensor = mnistnet(images)
            _, predicted = torch.max(outputs.data, 1)

            # Store predictions, true labels, and probabilities
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_proba.extend(outputs.cpu().numpy())

    print(classification_report(y_true, y_pred))
    print(f"Log Loss: {log_loss(y_true, y_pred_proba)}")

    torch.save(mnistnet.state_dict(), 'mnist_model.ckpt')
    
else:
    # Inference code starts
    mnist_ckpt = torch.load('mnist_model.ckpt')
    mnistnet.load_state_dict(mnist_ckpt)
