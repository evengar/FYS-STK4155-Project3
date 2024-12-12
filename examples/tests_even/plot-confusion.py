import torch
import torch.nn as nn
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from torchvision.io import read_image
from torch.utils.data import DataLoader
from pthree.image_utils import img_label_from_folder, split_imagedata, train_cnn, ConvNet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imageio.v2 import imread

import_dir = "examples/tests_even/cpics_data"
timestamp = "2024-12-09_1113"
img_size = 128
batch_size = 64
n_labels = 81

# get label dictionary
with open(f'{import_dir}/label_dict-{img_size}-{timestamp}.pkl', 'rb') as f:
    label_dict = pickle.load(f)
# shorten labels, and make reverse dict
label_dict_inv = {v: k.split(">")[-2]+">"+k.split(">")[-1] for k, v in label_dict.items()}
# latest run 
# learning rate 0.00015848931924611142, and lambda=0.001
# Accuracy: 0.7402912974357605, loss: 1.3585240528421494
# get model
input_dim = (3, img_size, img_size)
model = ConvNet(input_dim, n_labels, 64)
model.load_state_dict(torch.load(f"{import_dir}/best_model-{img_size}-{timestamp}.pt", weights_only=True))
model.eval()



# get test data
# saved with .npy extension for planktoscope grid (which is wrong)
test_set = torch.load(f"{import_dir}/test_set-{img_size}-{timestamp}.pt")

print(type(test_set))
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()

y_test = []
preds = []
n_correct = 0

img_row = 3
img_col = 4
fig, ax = plt.subplots(nrows=img_row, ncols=img_col)
n_plots = img_row*img_col
plot_files = []
correct_names = []
predicted_names = []
n_plotted = 0

with torch.no_grad():
    for x, y, files in test_dl:
        pred = model(x)
        loss = loss_fn(pred, y)
        pred_class = torch.argmax(pred, dim=1)
        is_correct = (
            pred_class == y
        ).float()
        y = y.numpy()
        pred_class = pred_class.numpy()

        for i in range(len(pred_class)):
            y_test.append(int(y[i]))
            preds.append(int(pred_class[i]))
            if y[i] != pred_class[i] and n_plotted < 25:
                plot_files.append(files[i])
                correct_names.append(label_dict_inv[y[i]])
                predicted_names.append(label_dict_inv[pred_class[i]])

        n_correct += sum(is_correct)

image_index = 0
for i in range(img_row):
    for j in range(img_col):
        img = imread(plot_files[image_index])
        title = f"Actual: {correct_names[image_index]}\nPredicted: {predicted_names[image_index]}"
        ax[i,j].imshow(img)
        ax[i,j].axis("off")
        ax[i,j].set_title(title, fontsize=5)
        image_index += 1
fig.tight_layout()
fig.savefig(f"examples/tests_even/figs/wrong-preds2-{timestamp}.pdf", bbox_inches="tight")



final_acc = float((n_correct / len(y_test)).numpy())

# print(y_test)
# print(preds)


y_labels = [label_dict_inv[val] for val in range(len(label_dict))]

print(label_dict)
print(label_dict_inv)

count_dict = {i:0 for i in range(14)}
for i in range(len(y_test)):
    count_dict[y_test[i]] += 1

print(count_dict)

cm = confusion_matrix(y_test, preds)
print(cm)
with open(f'{import_dir}/confusion-matrix-{img_size}-{timestamp}.pkl', 'wb') as f:
    pickle.dump(cm, f)
ConfusionMatrixDisplay(cm).plot()
plt.xticks(range(len(label_dict)), y_labels, rotation = 90, ha="right")
plt.yticks(range(len(label_dict)), y_labels)
plt.title(f"Test accuracy: {round(100*final_acc, 1)} %")
figure = plt.gcf() # get current figure
figure.set_size_inches(12, 12)
# when saving, specify the DPI
plt.savefig(f"examples/tests_even/figs/confusion-matrix-{timestamp}.pdf", bbox_inches="tight", dpi=200)

