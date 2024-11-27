import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy

from torch.utils.data import DataLoader
from pthree.image_utils import img_label_from_folder, split_imagedata, train_cnn, ConvNet

print("Cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# disable cudnn, see https://stackoverflow.com/questions/48445942/pytorch-training-with-gpu-gives-worse-error-than-training-the-same-thing-with-c
torch.backends.cudnn.enabled = False

torch.manual_seed(92348)


img_size = int(sys.argv[1])

file_list, labels, label_dict = img_label_from_folder(f"data/img/{img_size}/")

train_set, valid_set, test_set = split_imagedata(file_list, labels)

batch_size = 64

train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)


input_dim = (3, img_size, img_size)
output_channels=len(set(labels))

loss_fn = nn.CrossEntropyLoss()
lmbs = np.logspace(-5, 0, 6)
lrs = np.logspace(-4, 0, 4)

final_acc = np.zeros((len(lmbs), len(lrs)))
final_loss = np.ones((len(lmbs), len(lrs)))
num_epochs = 20
best_acc = 0
best_model = None

for i, lmb in enumerate(lmbs):
    for j, lr in enumerate(lrs):
        model = ConvNet(input_dim = input_dim, output_channels=output_channels, batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lmb)
        loss_train, loss_valid, acc_train, acc_valid = train_cnn(model, num_epochs, train_dl, valid_dl, optimizer=optimizer, device=device, loss_fn=loss_fn)
        acc = acc_valid[-1]
        loss = loss_valid[-1]
        final_acc[i, j] = acc
        final_loss[i, j] = loss
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)


