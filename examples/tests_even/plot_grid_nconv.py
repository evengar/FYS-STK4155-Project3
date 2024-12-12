import torch
import torch.nn as nn
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pthree.image_utils import img_label_from_folder, split_imagedata, train_cnn, ConvNet

def lambda_nconv_heatmap(mses, nconv, learning_rates,
                      lmb_label_res=3, lr_label_res=3, file=None):
    nconv_lab = nconv + 1
    lr_lab = ["{0:.2e}".format(x) for x in learning_rates]
    sns.heatmap(mses, annot=True, cbar_kws={'label': 'Validation accuracy'})
    plt.xticks(np.arange(
    len(learning_rates))[::lmb_label_res] + 0.5, lr_lab[::lmb_label_res]
    )
    plt.yticks(np.arange(
    len(nconv))[::lr_label_res] + 0.5, nconv_lab[::lr_label_res]
    )
    plt.xlabel(r"Learning rate $\eta$")
    plt.ylabel(r"Number of convolutional layers")
    
    if file is not None:
        plt.savefig(file)
    else:
        plt.show()

import_dir = "examples/tests_even/data_out"
timestamp = "2024-12-12_1108"
img_size = 128


lrs = np.load(f"{import_dir}/lrs-{timestamp}.npy")
nconv = np.arange(3)
accuracy = np.load(f"{import_dir}/accuracy-{img_size}-{timestamp}.npy")

lambda_nconv_heatmap(accuracy, nconv, lrs, 
                  lmb_label_res=2, lr_label_res=1, 
                  file = f"examples/tests_even/figs/gridsearch-nconv-{img_size}-{timestamp}.pdf")


# TODO evaluate model and show confusion matrix
# NOTE move to separate script, do for best model in the end
