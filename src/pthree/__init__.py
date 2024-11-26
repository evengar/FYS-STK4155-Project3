#!/usr/bin/env python
# -*- coding: utf-8 -*-


import importlib.metadata

# imports
from .plot import set_plt_params
from .image_utils import img_label_from_folder, split_imagedata, train_cnn, ImageDataset

__version__ = importlib.metadata.version(__package__)

# add function names
__all__ = [
    "set_plt_params",
    "img_label_from_folder",
    "split_imagedata",
    "train_cnn",
    "ImageDataset",

]
