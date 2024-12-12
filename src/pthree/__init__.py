#!/usr/bin/env python
# -*- coding: utf-8 -*-


import importlib.metadata

# imports
from .plot import set_plt_params
from .image_utils import img_label_from_folder, split_imagedata, train_cnn, ImageDataset
from .create_dataset import create_full_dataset, feature_selection_ecotaxa, feature_selection_dino, feature_selection_dino_pca
from .tree_clf import decision_tree_planktoscope, adaboost_planktoscope

__version__ = importlib.metadata.version(__package__)

# add function names
__all__ = [
    "set_plt_params",
    "img_label_from_folder",
    "split_imagedata",
    "train_cnn",
    "ImageDataset",
    "create_full_dataset",
    "feature_selection_ecotaxa", 
    "feature_selection_dino", 
    "feature_selection_dino_pca",
    "decision_tree_planktoscope", 
    "adaboost_planktoscope"
]
