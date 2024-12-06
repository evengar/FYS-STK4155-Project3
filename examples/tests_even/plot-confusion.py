import torch
import torch.nn as nn
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pthree.image_utils import img_label_from_folder, split_imagedata, train_cnn, ConvNet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import_dir = "examples/tests_even/data_out"
timestamp = "2024-12-02_0831"
img_size = 128
batch_size = 64

# get label dictionary
file_list, labels, label_dict = img_label_from_folder(f"data/img/{img_size}/")
print(label_dict)
for i in range(10):
    file_list, labels, label_dict = img_label_from_folder(f"data/img/{img_size}/")
    print(label_dict)
# for i in range(20):
#     file_list, labels, label_dict = img_label_from_folder(f"data/img/{img_size}/")
#     print(label_dict)

# get model
input_dim = (3, img_size, img_size)
model = ConvNet(input_dim, 14, 64)
model.load_state_dict(torch.load(f"examples/tests_even/data_out/best_model-{img_size}-{timestamp}.pt", weights_only=True))
model.eval()



# get test data
# saved with .npy extension (which is wrong)
test_set = torch.load(f"examples/tests_even/data_out/test_set-{img_size}-{timestamp}.npy")
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()

y_test = []
preds = []
n_correct = 0

with torch.no_grad():
    for x, y in test_dl:
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
        n_correct += sum(is_correct)

final_acc = float((n_correct / len(y_test)).numpy())

# print(y_test)
# print(preds)

label_dict_inv = {v: k for k, v in label_dict.items()}
y_labels = [label_dict_inv[val] for val in range(len(label_dict))]

print(label_dict)
print(label_dict_inv)

count_dict = {i:0 for i in range(14)}
for i in range(len(y_test)):
    count_dict[y_test[i]] += 1

print(count_dict)

cm = confusion_matrix(y_test, preds)
ConfusionMatrixDisplay(cm).plot()
plt.xticks(range(len(label_dict)), y_labels, rotation = 90)
plt.yticks(range(len(label_dict)), y_labels)
plt.title(f"Test accuracy: {round(100*final_acc, 1)} %")
plt.show()