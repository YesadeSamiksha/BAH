from scipy.io import loadmat

from config import DATASET_PATH

print("Loading dataset...")

data = loadmat(DATASET_PATH)

print("\nKeys:")
print(data.keys())

print("\nShapes:")
print("PAN:", data["PAN_IMAGES"].shape)
print("MUL:", data["MUL_IMAGES"].shape)
print("LABELS:", data["LAND_COVER_TYPES"].shape)