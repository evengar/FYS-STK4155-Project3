import numpy as np

import_dir = "examples/tests_even/cpics_data"
timestamp1 = "2024-12-06_0945"
timestamp2 = "2024-12-09_1113"
img_size = 128


lrs1 = np.load(f"{import_dir}/lrs-{timestamp1}.npy")
lmbs1 = np.load(f"{import_dir}/lmbs-{timestamp1}.npy")
lrs2 = np.load(f"{import_dir}/lrs-{timestamp2}.npy")
lmbs2 = np.load(f"{import_dir}/lmbs-{timestamp2}.npy")
accuracy1 = np.load(f"{import_dir}/accuracy-{img_size}-{timestamp1}.npy")
accuracy2 = np.load(f"{import_dir}/accuracy-{img_size}-{timestamp2}.npy")

accuracy = np.column_stack((accuracy2, accuracy1))
lrs = np.concat((lrs2, lrs1))

print(lrs)
print(lmbs1, lmbs2)
print(accuracy)

np.save(f"{import_dir}/accuracy-{img_size}-{timestamp2}a.npy", accuracy)
np.save(f"{import_dir}/lmbs-{timestamp2}a.npy", lmbs2)
np.save(f"{import_dir}/lrs-{timestamp2}a.npy", lrs)