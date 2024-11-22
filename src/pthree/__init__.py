#!/usr/bin/env python
# -*- coding: utf-8 -*-


import importlib.metadata

# imports
from .plot import set_plt_params

__version__ = importlib.metadata.version(__package__)

# add function names
__all__ = [
    "set_plt_params"
]
