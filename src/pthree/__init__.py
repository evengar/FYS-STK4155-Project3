#!/usr/bin/env python
# -*- coding: utf-8 -*-


import importlib.metadata

# imports
from .plot import set_plt_params
from .image_utils import img_label_from_folder, split_imagedata, train_cnn, ImageDataset
from .create_dataset import feature_selection_ecotaxa, feature_selection_dino, feature_selection_dino_pca, feature_selection_dino_cpics
from .tree_clf import decision_tree_planktoscope, adaboost_planktoscope

__version__ = importlib.metadata.version(__package__)

# add function names
__all__ = [
    "set_plt_params",
    "img_label_from_folder",
    "split_imagedata",
    "train_cnn",
    "ImageDataset",
    "feature_selection_ecotaxa", 
    "feature_selection_dino", 
    "feature_selection_dino_pca",
    "feature_selection_dino_cpics",
    "decision_tree_planktoscope", 
    "adaboost_planktoscope"
]
