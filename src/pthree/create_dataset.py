import numpy as np
import pandas as pd
import git
import os


def create_full_dataset(directory, save_as):
    """Function to combine tsv files into a full dataset."""
    path = f"{directory}{save_as}.csv"
    if os.path.isfile(path):
        print("Full ecotaxa dataset exist!")
        return
    
    dataset = []

    for file in os.scandir(directory):
        if file.is_file():
            filename = os.path.join(directory, file)
            data = pd.read_csv(filename, sep="\t")
            data.drop([0], inplace=True)
            dataset.append(data)
        else:
            continue
    
    full_dataset = pd.concat(dataset, ignore_index=True)
    full_dataset.to_csv(f"{directory}{save_as}.csv")


def feature_selection_ecotaxa(path_file):
    """Work in progress, need to decide on features to use."""
    dataset = pd.read_csv(path_file, index_col=0)

    dataset["target_name"] = dataset.apply(lambda x: x["img_file_name"].split("/")[0], axis=1)
    labels = np.unique(dataset.apply(lambda x: x["target_name"], axis=1)).tolist()
    dataset["target"] = dataset.apply(lambda x: labels.index(x["target_name"]), axis=1)

    start_idx = dataset.columns.to_list().index("object_label")
    end_idx = dataset.columns.to_list().index("object_date_end")
    features = dataset.columns.to_list()[start_idx:end_idx]

    X = dataset[[c for c in dataset.columns if c in features]].to_numpy()
    y = dataset["target"].to_numpy()

    return X, y


if __name__ == '__main__':
    PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
    directory = f"{PATH_TO_ROOT}/data/metadata/"

    create_full_dataset(directory, "ecotaxa_full")
    
    path_file = f"{directory}ecotaxa_full.csv"
    X, y = feature_selection_ecotaxa(path_file)