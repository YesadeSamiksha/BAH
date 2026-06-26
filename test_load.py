from scipy.io import loadmat

print("Loading dataset...")

data = loadmat("data/DSRSID.mat")

print("\nKeys:")
print(data.keys())

print("\nShapes:")
print("PAN:", data["PAN_IMAGES"].shape)
print("MUL:", data["MUL_IMAGES"].shape)
print("LABELS:", data["LAND_COVER_TYPES"].shape)