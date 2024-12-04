import numpy as np
import pandas as pd
import git
import os

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir


def create_full_dataset(directory, save_as):
    """Function to combine tsv files into a full dataset."""
    dataset = []

    for file in os.scandir(directory):
        if file.is_file():
            filename = os.path.join(directory, file)
            dataset.append(pd.read_csv(filename, sep="\t"))
        else:
            continue
    
    full_dataset = pd.concat(dataset)
    full_dataset.to_csv(f"{directory}{save_as}.csv")


def preprocess_ecotaxa(path_file):
    """Work in progress, need to decide on features to use."""
    dataset = pd.read_csv(path_file)

    print(dataset.head())


if __name__ == '__main__':
    directory = f"{PATH_TO_ROOT}/data/metadata/"

    create_full_dataset(directory, "ecotaxa_full")
    
    path_file = f"{directory}ecotaxa_full.csv"
    preprocess_ecotaxa(path_file)