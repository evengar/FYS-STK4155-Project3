import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys

from torch.utils.data import DataLoader
from pthree.image_utils import img_label_from_folder, split_imagedata, train_cnn

print("Cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

img_size = int(sys.argv[1])

file_list, labels, label_dict = img_label_from_folder(f"data/img/{img_size}/")

train_set, valid_set, test_set = split_imagedata(file_list, labels)

batch_size = 16

train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)

class ConvNet(nn.Module):
    def __init__(self, input_dim, output_channels, batch_size = 16):
        super().__init__()

        self.feature_extractor = nn.Sequential(

            nn.Conv2d(
                in_channels=3, out_channels=32,
                kernel_size=5, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=5, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=5, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        channels, h, w = input_dim
        x = torch.ones((batch_size, channels, h, w))
        out_dim = self.feature_extractor(x).shape
        #print(out_dim)
        #num_features_before_fcnn = functools.reduce(operator.mul, list(self.feature_extractor(torch.rand(batch_size, *input_dim)).shape))
        self.classifier = nn.Sequential(
            nn.Linear(out_dim[1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 400),
            nn.ReLU(),
            nn.Linear(400, output_channels)
        )
        #print(self.classifier(self.feature_extractor(x)).shape)


    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.classifier(out)
        return out

input_dim = (3, img_size, img_size)
output_channels=len(set(labels))
print(output_channels)

model = ConvNet(input_dim = input_dim, output_channels=output_channels, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# run network
torch.manual_seed(432987)
num_epochs = 2
hist0, hist1, hist2, hist3 = train_cnn(model, num_epochs, train_dl, valid_dl, optimizer=optimizer, device=device, loss_fn=loss_fn)

hist2 = [h.cpu() for h in hist2]
hist3 = [h.cpu() for h in hist3]


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
plt.savefig(f"examples/tests_even/figs/CNN-test-{img_size}.pdf")