
# %matplotlib inline
import random
import pandas as pd
import numpy as np
from array import array
from os.path import join
import struct
import matplotlib.pyplot as plt


# Set file paths based on added MNIST Datasets
#
input_path = '../archive/'
train_im_path = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
train_lab_path = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_im_path = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_lab_path = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=None)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1


def im_lab_unpack(im_path, lab_path):
    labels = []
    with open(lab_path, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))        # 8 bit magic number
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())        
        
    with open(im_path, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())        

    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img            
        
    return images, labels

x_train, y_train = im_lab_unpack(train_im_path, train_lab_path)
x_test, y_test = im_lab_unpack(test_im_path, test_lab_path)

images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

show_images(images_2_show, titles_2_show)
