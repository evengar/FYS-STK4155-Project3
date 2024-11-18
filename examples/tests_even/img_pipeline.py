import pathlib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from sklearn.model_selection import train_test_split

imgdir_path = pathlib.Path("data/img/128/")
file_list = sorted([str(path) for path in imgdir_path.rglob('*.jpg')])
#print(file_list)

category = [path.split("/")[3] for path in file_list]

label_dict = {cat:i for i, cat in enumerate(set(category))}
labels = [label_dict[cat] for cat in category]

img_train_, img_test, label_train_, label_test = train_test_split(file_list, labels, test_size=0.2)
img_train, img_valid, label_train, label_valid = train_test_split(img_train_, label_train_, test_size=0.25) # = 0.2 of full data set
print(len(img_train))

class ImageDataset(Dataset):
    def __init__(self, file_list, labels):
        self.file_list = file_list
        self.labels = labels
    def __getitem__(self, index):
        file = self.file_list[index]
        image = read_image(file)
        image = image / 255 # convert to (minmax) float for compatibility
        label = self.labels[index]
        return image, label
    def __len__(self):
        return len(self.labels)
    def dim(self):
        return self[0][0].shape

train_set = ImageDataset(img_train, label_train)
valid_set = ImageDataset(img_valid, label_valid)
test_set = ImageDataset(img_test, label_test)

print(train_set.dim())

batch_size = 16

train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)


# # Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img_ = train_features[0].squeeze()
# label = train_labels[0]
# img = img_.permute(1, 2, 0)
# plt.imshow(img)
# plt.title(f"{label}")
# plt.show()
# print(f"Label: {label}")

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

channels, h, w = test_set.dim()
x = torch.ones((batch_size, channels, h, w))
out_dim = model(x).shape

model.add_module('fc1', nn.Linear(out_dim[1], 1024))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(p=0.5))
model.add_module('fc2', nn.Linear(1024, len(set(category))))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, num_epochs, train_dl, valid_dl):
    """
    Train the CNN, from Raschka et al"""
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (
                torch.argmax(pred, dim=1) == y_batch
                ).float()
            accuracy_hist_train[epoch] += is_correct.sum()
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0)
                is_correct = (
                    torch.argmax(pred, dim=1) == y_batch
                ).float()
                accuracy_hist_valid[epoch] += is_correct.sum()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch+1} accuracy: '
            f'{accuracy_hist_train[epoch]:.4f} val_accuracy: '
            f'{accuracy_hist_valid[epoch]:.4f}')
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

torch.manual_seed(1)
num_epochs = 20
hist = train(model, num_epochs, train_dl, valid_dl)

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
plt.show()