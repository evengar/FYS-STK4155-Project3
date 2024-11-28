import torch
import torch.nn as nn
import numpy as np
import sys
import copy
import time
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt

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

torch.manual_seed(92348)


img_size = 224

file_list, labels, label_dict = img_label_from_folder(f"data/img/{img_size}/")

normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                        std  = [0.229, 0.224, 0.225])

train_set, valid_set, test_set = split_imagedata(file_list, labels, transform=normalize)

# save the test set for later assessing final test metrics on the best model
#torch.save(test_set, f"examples/tests_even/data_out/test_set-{img_size}-{timestamp}.npy")

batch_size = 64
output_channels=len(set(labels))

train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# freeze all model parameters
for param in model.parameters():
    param.requires_grad = False

# new final layer with 16 classes
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, output_channels)

torch.manual_seed(432987)
num_epochs = 20
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
plt.savefig(f"examples/tests_even/figs/CNN-resnet50-{img_size}.pdf")


# lmbs = np.logspace(-10, 0, 11)
# lrs = np.logspace(-7, 0, 8)
# np.save(f"examples/tests_even/data_out/lrs-{timestamp}.npy", lrs)
# np.save(f"examples/tests_even/data_out/lmbs-{timestamp}.npy", lmbs)

# final_acc = np.zeros((len(lmbs), len(lrs)))
# final_loss = np.ones((len(lmbs), len(lrs)))
# num_epochs = 20
# best_acc = 0

# for i, lmb in enumerate(lmbs):
#     for j, lr in enumerate(lrs):
#         model = ConvNet(input_dim = input_dim, output_channels=output_channels, batch_size=batch_size)
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lmb)
#         print(f"Running model for learning rate={lr}, lambda={lmb}")
#         loss_train, loss_valid, acc_train, acc_valid = train_cnn(model, num_epochs, train_dl, valid_dl, optimizer=optimizer, device=device, loss_fn=loss_fn)
#         acc = acc_valid[-1]
#         loss = loss_valid[-1]
#         final_acc[i, j] = acc
#         final_loss[i, j] = loss
#         if acc > best_acc:
#             best_acc = acc
#             best_loss = loss
#             best_model = copy.deepcopy(model)
#             best_lmb = lmb
#             best_lr = lr

# print(f"Best model found for learning rate={best_lr}, and lambda={best_lmb}")
# print(f"Accuracy: {best_acc}, loss: {best_loss}")

# np.save(f"examples/tests_even/data_out/accuracy-{img_size}-{timestamp}.npy", final_acc)
# np.save(f"examples/tests_even/data_out/loss-{img_size}-{timestamp}.npy", final_loss)

# best_model = best_model.to("cpu")
# torch.save(best_model.state_dict(), f"examples/tests_even/data_out/best_model-{img_size}-{timestamp}.pt")