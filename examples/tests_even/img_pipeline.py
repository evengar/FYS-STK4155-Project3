import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import sys

from torch.utils.data import DataLoader
from pthree.image_utils import img_label_from_folder, split_imagedata, train_cnn, ConvNet

print("Cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# disable cudnn, see https://stackoverflow.com/questions/48445942/pytorch-training-with-gpu-gives-worse-error-than-training-the-same-thing-with-c
torch.backends.cudnn.enabled = False

img_size = int(sys.argv[1])

file_list, labels, label_dict = img_label_from_folder(f"data/img/{img_size}/")

normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                        std  = [0.229, 0.224, 0.225])

train_set, valid_set, test_set = split_imagedata(file_list, labels, transform=normalize)

batch_size = 16

train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)


input_dim = (3, img_size, img_size)
output_channels=len(set(labels))
print(output_channels)

model = ConvNet(input_dim = input_dim, output_channels=output_channels, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# run network
torch.manual_seed(432987)
num_epochs = 100
hist0, hist1, hist2, hist3 = train_cnn(model, num_epochs, train_dl, valid_dl, optimizer=optimizer, device=device, loss_fn=loss_fn)


x_arr = np.arange(len(hist0)) + 1
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist0, '-o', label='Train loss')
ax.plot(x_arr, hist1, '--<', label='Validation loss')
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist2, '-o', label='Train acc.')
ax.plot(x_arr, hist3, '--<',
label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
plt.savefig(f"examples/tests_even/figs/CNN-test-epochs{num_epochs}-{img_size}.pdf")