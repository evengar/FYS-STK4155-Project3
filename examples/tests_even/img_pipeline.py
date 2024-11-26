import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from pthree.image_utils import img_label_from_folder, split_imagedata, train_cnn

img_size = 64

file_list, labels, label_dict = img_label_from_folder(f"data/img/{img_size}/")

train_set, valid_set, test_set = split_imagedata(file_list, labels)

batch_size = 16

train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Specify model
# Convolutional layer
model = nn.Sequential()
model.add_module(
    "conv1",
    nn.Conv2d(
        in_channels=3, out_channels=32,
        kernel_size=5, padding=2
    )
)
model.add_module("relu1", nn.ReLU())
model.add_module("pool1", nn.MaxPool2d(kernel_size=2))
model.add_module("flatten", nn.Flatten())

# get correct dims for fully-connected layer
channels, h, w = test_set.dim()
x = torch.ones((batch_size, channels, h, w))
out_dim = model(x).shape

# add fully-connected layer
model.add_module('fc1', nn.Linear(out_dim[1], 1024))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(p=0.5))
model.add_module('fc2', nn.Linear(1024, len(set(labels))))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# run network
torch.manual_seed(432987)
num_epochs = 20
hist = train_cnn(model, num_epochs, train_dl, valid_dl, optimizer=optimizer, loss_fn=loss_fn)

x_arr = np.arange(len(hist[0])) + 1
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0], '-o', label='Train loss')
ax.plot(x_arr, hist[1], '--<', label='Validation loss')
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2], '-o', label='Train acc.')
ax.plot(x_arr, hist[3], '--<',
label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
plt.savefig(f"examples/tests_even/figs/CNN-test-{img_size}.pdf")