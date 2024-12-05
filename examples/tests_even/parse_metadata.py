import numpy as np

file = "data/metadata/cpics_full.tsv"

filenames = []
labels = []

with open(file) as f:
    lines = f.readlines()
    for i in range(2, len(lines)):
        vals = lines[i].split("\t")
        filename = vals[0].strip('"')
        label = vals[16].strip('"')
        if label == "temporary>t003":
            print(filename, label)
            continue
        filenames.append(filename)
        labels.append(label)

filenames = np.array(filenames)
labels = np.array(labels)

# print(filenames[:4])
# print(labels[:4])

np.save("examples/tests_even/cpics_data/cpics_filenames.npy", filenames)
np.save("examples/tests_even/cpics_data/cpics_labels.npy", labels)
