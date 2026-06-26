import h5py

from config import DATASET_PATH

with h5py.File(DATASET_PATH, "r") as f:

    print("Keys:")
    print(list(f.keys()))

    print("\nShapes:")

    for key in f.keys():
        print(key, f[key].shape)