import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle 


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
print(output_channels)

model = ConvNet(input_dim = input_dim, output_channels=output_channels, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 1e-5)

torch.manual_seed(53789)
num_epochs = 100
hist0, hist1, hist2, hist3 = train_cnn(model, num_epochs, train_dl, valid_dl, optimizer=optimizer, device=device, loss_fn=loss_fn)

model = model.to("cpu")
torch.save(model.state_dict(), f"examples/tests_even/cpics_data/best_model-{img_size}-{timestamp}.pt")

print("Model finished running")
print(f"Final train loss: {hist0[-1]}")
print(f"Final validation loss: {hist1[-1]}")
print(f"Final train accuracy: {hist2[-1]}")
print(f"Final validation accuracy: {hist3[-1]}")

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
plt.savefig(f"examples/tests_even/figs/cpics-CNN-epochs{num_epochs}-{img_size}.pdf")
