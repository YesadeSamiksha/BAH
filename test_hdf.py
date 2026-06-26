import h5py

with h5py.File("data/DSRSID.mat", "r") as f:

    print("Keys:")
    print(list(f.keys()))

    print("\nShapes:")

    for key in f.keys():
        print(key, f[key].shape)