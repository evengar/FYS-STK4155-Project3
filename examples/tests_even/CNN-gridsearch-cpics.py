import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle 
import copy


from torch.utils.data import DataLoader
from pthree.image_utils import split_imagedata, train_cnn, ConvNet

print("Cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# disable cudnn, see https://stackoverflow.com/questions/48445942/pytorch-training-with-gpu-gives-worse-error-than-training-the-same-thing-with-c
torch.backends.cudnn.enabled = False

t = time.localtime()
timestamp=time.strftime('%Y-%m-%d_%H%M', t)

img_size = 128
input_dir = "data/cpics/128/"


file_list = np.load("examples/tests_even/cpics_data/cpics_filenames.npy")
file_list = input_dir + file_list

print(file_list[:4])

label_list = np.load("examples/tests_even/cpics_data/cpics_labels.npy")

label_dict = {cat:i for i, cat in enumerate(set(label_list))}
labels = [label_dict[cat] for cat in label_list]

with open(f'examples/tests_even/cpics_data/label_dict-{img_size}-{timestamp}.pkl', 'wb') as f:
    pickle.dump(label_dict, f)

normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                        std  = [0.229, 0.224, 0.225])


train_set, valid_set, test_set = split_imagedata(file_list, labels, transform=normalize, test_size=0.25, valid_size=0.25)

# save all sets to have the option to keep training
torch.save(train_set, f"examples/tests_even/cpics_data/train_set-{img_size}-{timestamp}.pt")
torch.save(valid_set, f"examples/tests_even/cpics_data/valid_set-{img_size}-{timestamp}.pt")
torch.save(test_set, f"examples/tests_even/cpics_data/test_set-{img_size}-{timestamp}.pt")

batch_size = 64

train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)


input_dim = (3, img_size, img_size)
output_channels=len(set(labels))

model = ConvNet(input_dim = input_dim, output_channels=output_channels, batch_size=batch_size)

# initialize small grid of lmbs and lrs
loss_fn = nn.CrossEntropyLoss()

lmbs = np.array([1e-4, 1e-3, 1e-2])
lrs = np.array([1e-5, 1e-4])
np.save(f"examples/tests_even/cpics_data/lrs-{timestamp}.npy", lrs)
np.save(f"examples/tests_even/cpics_data/lmbs-{timestamp}.npy", lmbs)

final_acc = np.zeros((len(lmbs), len(lrs)))
final_loss = np.ones((len(lmbs), len(lrs)))
num_epochs = 12
best_acc = 0

for i, lmb in enumerate(lmbs):
    for j, lr in enumerate(lrs):
        model = ConvNet(input_dim = input_dim, output_channels=output_channels, batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lmb)
        print(f"Running model for learning rate={lr}, lambda={lmb}")
        loss_train, loss_valid, acc_train, acc_valid = train_cnn(model, num_epochs, train_dl, valid_dl, optimizer=optimizer, device=device, loss_fn=loss_fn)
        acc = acc_valid[-1]
        loss = loss_valid[-1]
        final_acc[i, j] = acc
        final_loss[i, j] = loss
        if acc > best_acc:
            best_acc = acc
            best_loss = loss
            best_model = copy.deepcopy(model)
            best_lmb = lmb
            best_lr = lr

print(f"Best model found for learning rate={best_lr}, and lambda={best_lmb}")
print(f"Accuracy: {best_acc}, loss: {best_loss}")

np.save(f"examples/tests_even/cpics_data/accuracy-{img_size}-{timestamp}.npy", final_acc)
np.save(f"examples/tests_even/cpics_data/loss-{img_size}-{timestamp}.npy", final_loss)

best_model = best_model.to("cpu")
torch.save(best_model.state_dict(), f"examples/tests_even/cpics_data/best_model-{img_size}-{timestamp}.pt")
