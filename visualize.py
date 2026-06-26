import h5py
import matplotlib.pyplot as plt

from config import DATASET_PATH

with h5py.File(DATASET_PATH, "r") as f:

    pan = f["PAN_IMAGES"]
    mul = f["MUL_IMAGES"]

    pan_img = pan[0][0]
    mul_img = mul[0][:3]

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(pan_img, cmap="gray")
plt.title("PAN Image")

plt.subplot(1,2,2)
plt.imshow(mul_img.transpose(1,2,0))
plt.title("MUL RGB")

plt.show()