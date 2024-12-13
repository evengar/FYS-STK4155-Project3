import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, Normalize
from pthree.image_utils import img_label_from_folder, split_imagedata, train_cnn, ConvNet
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import_dir = "examples/tests_even/cpics_data"
timestamp = "2024-12-09_1113"
img_size = 128
batch_size = 64
n_labels = 81

with open("examples/tests_even/cpics_data/confusion-matrix-128-2024-12-06_0945.pkl", 'rb') as f:
    cm = pickle.load(f)

with open(f'{import_dir}/label_dict-{img_size}-{timestamp}.pkl', 'rb') as f:
    label_dict = pickle.load(f)

test_set = torch.load(f"{import_dir}/test_set-{img_size}-{timestamp}.pt")

count_dict = {i:0 for i in range(n_labels)}
for img, label, file in test_set:
    count_dict[label] += 1

print(count_dict)

label_dict_inv = {v: k for k, v in label_dict.items()}
y_labels = np.array([label_dict_inv[val] for val in range(len(label_dict)) if count_dict[val] > 0])


for i, lab in enumerate(y_labels):
    levels = lab.split(">")
    new_lab = levels[-2] + ">" + levels[-1]
    y_labels[i] = new_lab
#print(y_labels)

filter_indices = np.where(np.sum(cm, axis=1) > 10)

y_labels = y_labels[filter_indices]
cm = cm[np.ix_(filter_indices[0], filter_indices[0])]

final_acc = 0.75

count_is_zero = cm == 0
logcm = np.log10(cm + (count_is_zero * 0.1))

sorted_indices = np.argsort(y_labels)
y_labels = [y_labels[i] for i in sorted_indices]

# Sort the array rows and columns based on sorted indices
logcm = logcm[np.ix_(sorted_indices, sorted_indices)]


sns.heatmap(logcm, annot=True, annot_kws={"fontsize":4}, cbar_kws={'label': r'$\log_{10}(\text{count})$'}, cmap="mako", square = True)
plt.xticks(np.arange(len(y_labels)) + 0.5, y_labels, rotation = 90, fontsize=4)
plt.yticks(np.arange(len(y_labels)) + 0.5, y_labels, rotation = 0, fontsize=4)
plt.title(f"CPICS CNN, accuracy {final_acc}")
figure = plt.gcf() # get current figure
figure.set_size_inches(12, 12)
# when saving, specify the DPI
plt.savefig(f"examples/tests_even/figs/cpics-confusion_{timestamp}_above10_log10.pdf", bbox_inches="tight")

plt.clf()
cm_norm = np.round(cm / cm.sum(axis=1)[:, np.newaxis] * 100, 1)
cm_norm_labels = np.empty_like(cm_norm)
annot_mask = cm_norm > 0.0099
cm_norm_labels[annot_mask] = cm_norm[annot_mask]
sns.heatmap(cm_norm, annot=cm_norm_labels, annot_kws={"fontsize":6}, cbar=False, cmap="mako", square = True, fmt=".1f")
plt.xticks(np.arange(len(y_labels)) + 0.5, y_labels, rotation = 90, fontsize=8)
plt.yticks(np.arange(len(y_labels)) + 0.5, y_labels, rotation = 0, fontsize=8)
plt.title(f"CPICS CNN, accuracy {final_acc}")
figure = plt.gcf() # get current figure
figure.set_size_inches(12, 12)
print("saving fig")
# when saving, specify the DPI
plt.savefig(f"examples/tests_even/figs/cpics-confusion_{timestamp}_above10_rel.pdf", bbox_inches="tight")