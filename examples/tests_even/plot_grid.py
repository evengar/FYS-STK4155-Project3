import torch
import torch.nn as nn
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pthree.image_utils import img_label_from_folder, split_imagedata, train_cnn, ConvNet

def lambda_lr_heatmap(mses, lmbs, learning_rates,
                      lmb_label_res=3, lr_label_res=3, file=None):
    lmb_lab = ["{0:.2e}".format(x) for x in lmbs]
    lr_lab = ["{0:.2e}".format(x) for x in learning_rates]
    sns.heatmap(mses, annot=True)
    plt.xticks(np.arange(
    len(learning_rates))[::lmb_label_res] + 0.5, lr_lab[::lmb_label_res]
    )
    plt.yticks(np.arange(
    len(lmbs))[::lr_label_res] + 0.5, lmb_lab[::lr_label_res]
    )
    plt.xlabel(r"Learning rate $\eta$")
    plt.ylabel(r"$\lambda$")
    if file is not None:
        plt.savefig(file)
    else:
        plt.show()

import_dir = "examples/tests_even/cpics_data"
timestamp = "2024-12-09_1113a"
img_size = 128


lrs = np.load(f"{import_dir}/lrs-{timestamp}.npy")
lmbs = np.load(f"{import_dir}/lmbs-{timestamp}.npy")
accuracy = np.load(f"{import_dir}/accuracy-{img_size}-{timestamp}.npy")

lambda_lr_heatmap(accuracy, lmbs, lrs, 
                  lmb_label_res=1, lr_label_res=1, 
                  file = f"examples/tests_even/figs/gridsearch-{img_size}-{timestamp}.pdf")


# TODO evaluate model and show confusion matrix
# NOTE move to separate script, do for best model in the end
