import h5py

mat_path = r"D:\ISRO_Hackathon\data\DSRSID.mat"

with h5py.File(mat_path, "r") as f:

    print(type(f["PAN_IMAGES"]))
    print(type(f["MUL_IMAGES"]))
    print(type(f["LAND_COVER_TYPES"]))

    if hasattr(f["PAN_IMAGES"], "shape"):
        print("PAN shape:", f["PAN_IMAGES"].shape)

    if hasattr(f["MUL_IMAGES"], "shape"):
        print("MUL shape:", f["MUL_IMAGES"].shape)

    if hasattr(f["LAND_COVER_TYPES"], "shape"):
        print("Labels shape:", f["LAND_COVER_TYPES"].shape)