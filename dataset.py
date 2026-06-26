import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class DSRSIDDataset(Dataset):
    """
    PyTorch Dataset for the DSRSID dataset.
    Loads data lazily using h5py to avoid loading the entire dataset into RAM.
    Supports a custom subset of indices for stratified sampling and reproducibility.
    """
    def __init__(self, file_path, indices=None, limit=5000, transform_pan=None, transform_mul=None):
        """
        Args:
            file_path (str): Path to the DSRSID .mat file.
            indices (list or numpy.ndarray, optional): Specific indices to use. 
                If provided, overrides 'limit'.
            limit (int): Number of samples to use if indices are not provided.
            transform_pan (callable, optional): Transforms to apply to PAN images.
            transform_mul (callable, optional): Transforms to apply to MUL images.
        """
        self.file_path = file_path
        self.indices = indices
        self.transform_pan = transform_pan
        self.transform_mul = transform_mul
        self.file = None  # To be initialized lazily in __getitem__ for multiprocessing compatibility

        # Validate key existence and shapes
        with h5py.File(self.file_path, "r") as f:
            if "PAN_IMAGES" not in f or "MUL_IMAGES" not in f or "LAND_COVER_TYPES" not in f:
                raise KeyError("Dataset file must contain 'PAN_IMAGES', 'MUL_IMAGES', and 'LAND_COVER_TYPES'")
            total_samples = f["PAN_IMAGES"].shape[0]

        if self.indices is not None:
            self.indices = np.array(self.indices)
            self.length = len(self.indices)
        else:
            self.length = min(total_samples, limit)
            self.indices = np.arange(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of bounds")

        # Lazily open HDF5 file
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")

        # Map to the subset index
        target_idx = self.indices[idx]

        # 1. Read PAN Image (1, 256, 256)
        pan_img = self.file["PAN_IMAGES"][target_idx]
        pan_2d = pan_img[0]  # Get the 2D channel
        pan_pil = Image.fromarray(pan_2d, mode='L').convert('RGB')  # Single channel to 3 channels

        # 2. Read MUL Image (4, 64, 64)
        mul_img = self.file["MUL_IMAGES"][target_idx]
        mul_rgb_np = mul_img[:3]  # Extract first 3 channels for RGB
        mul_rgb_np = np.transpose(mul_rgb_np, (1, 2, 0))  # Convert (3, 64, 64) to (64, 64, 3)
        mul_pil = Image.fromarray(mul_rgb_np, mode='RGB')

        # 3. Read label
        label = self.file["LAND_COVER_TYPES"][0, target_idx]

        # Default transforms if not provided
        from torchvision import transforms
        default_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Apply transforms
        pan_tensor = self.transform_pan(pan_pil) if self.transform_pan else default_transform(pan_pil)
        mul_tensor = self.transform_mul(mul_pil) if self.transform_mul else default_transform(mul_pil)

        return pan_tensor, mul_tensor, float(label)

    def get_visualization_images(self, idx):
        """
        Retrieves unnormalized PIL images of size 224x224 for visualization purposes.
        
        Args:
            idx (int): Dataset index (0 to length-1)
            
        Returns:
            pan_pil (PIL.Image): Resized PAN RGB image.
            mul_pil (PIL.Image): Resized MUL RGB image.
            label (float): Land cover class label.
        """
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of bounds")

        if self.file is None:
            self.file = h5py.File(self.file_path, "r")

        target_idx = self.indices[idx]

        # Load raw images
        pan_img = self.file["PAN_IMAGES"][target_idx]
        pan_2d = pan_img[0]
        pan_pil = Image.fromarray(pan_2d, mode='L').convert('RGB')
        # Resize to 224x224 using Bilinear resampling for visualization
        pan_pil_resized = pan_pil.resize((224, 224), Image.Resampling.BILINEAR)

        mul_img = self.file["MUL_IMAGES"][target_idx]
        mul_rgb_np = mul_img[:3]
        mul_rgb_np = np.transpose(mul_rgb_np, (1, 2, 0))
        mul_pil = Image.fromarray(mul_rgb_np, mode='RGB')
        mul_pil_resized = mul_pil.resize((224, 224), Image.Resampling.BILINEAR)

        label = self.file["LAND_COVER_TYPES"][0, target_idx]

        return pan_pil_resized, mul_pil_resized, float(label)

    def close(self):
        """Closes the HDF5 file handle if it is open."""
        if self.file is not None:
            self.file.close()
            self.file = None
