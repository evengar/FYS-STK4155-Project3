import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import sys
import copy
import time
import pickle 

from torch.utils.data import DataLoader
from pthree.image_utils import img_label_from_folder, split_imagedata, train_cnn, ConvNet

t = time.localtime()
timestamp=time.strftime('%Y-%m-%d_%H%M', t)

print("Cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# disable cudnn, see https://stackoverflow.com/questions/48445942/pytorch-training-with-gpu-gives-worse-error-than-training-the-same-thing-with-c
torch.backends.cudnn.enabled = False

torch.manual_seed(97975)


img_size = 128

file_list, labels, label_dict = img_label_from_folder(f"data/img/{img_size}/")

# SAVE THE LABEL DICT FFS
with open(f'examples/tests_even/data_out/label_dict-{img_size}-{timestamp}.pkl', 'wb') as f:
    pickle.dump(label_dict, f)

normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                        std  = [0.229, 0.224, 0.225])

train_set, valid_set, test_set = split_imagedata(file_list, labels, transform=normalize)
# save the test set for later assessing final test metrics on the best model
torch.save(test_set, f"examples/tests_even/data_out/test_set-{img_size}-{timestamp}.npy")

batch_size = 64

train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)


input_dim = (3, img_size, img_size)
output_channels=len(set(labels))

loss_fn = nn.CrossEntropyLoss()
lrs = np.logspace(-5, -3, 6)
np.save(f"examples/tests_even/data_out/lrs-{timestamp}.npy", lrs)

final_acc = np.zeros((3, len(lrs)))
final_loss = np.ones((3, len(lrs)))
num_epochs = 30
best_acc = 0

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
            nn.Dropout(p=0.5),
            nn.Linear(1024, 400),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(400, output_channels)
        )

feature_extractor_list = [
    nn.Sequential(

            nn.Conv2d(
                in_channels=3, out_channels=32,
                kernel_size=5, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        ),
        nn.Sequential(

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
            nn.Flatten()
        ),
        nn.Sequential(

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
]

for i, feature_extr in enumerate(feature_extractor_list):
    for j, lr in enumerate(lrs):
        model = ConvNet(input_dim = input_dim, output_channels=output_channels, batch_size=batch_size)
        model.feature_extractor = feature_extr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print(f"Running model for learning rate={lr}, number of convolutional layers={i+1}")
        loss_train, loss_valid, acc_train, acc_valid = train_cnn(model, num_epochs, train_dl, valid_dl, optimizer=optimizer, device=device, loss_fn=loss_fn)
        acc = acc_valid[-1]
        loss = loss_valid[-1]
        final_acc[i, j] = acc
        final_loss[i, j] = loss
        if acc > best_acc:
            best_acc = acc
            best_loss = loss
            best_model = copy.deepcopy(model)
            n_conv = i+1
            best_lr = lr

print(f"Best model found for learning rate={best_lr}, and n_conv={n_conv}")
print(f"Accuracy: {best_acc}, loss: {best_loss}")

np.save(f"examples/tests_even/data_out/accuracy-{img_size}-{timestamp}.npy", final_acc)
np.save(f"examples/tests_even/data_out/loss-{img_size}-{timestamp}.npy", final_loss)

best_model = best_model.to("cpu")
torch.save(best_model.state_dict(), f"examples/tests_even/data_out/best_model-{img_size}-{timestamp}.pt")
