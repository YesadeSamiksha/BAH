import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from tqdm import tqdm

from dataset import DSRSIDDataset

def main():
    # 1. Compute and save stratified indices for reproducibility
    # Dataset has 8 classes, each has exactly 10,000 contiguous samples.
    # We sample the first 625 images from each class (8 * 625 = 5000 samples total).
    print("Computing stratified subset indices...")
    subset_indices = []
    num_classes = 8
    samples_per_class = 10000
    subsample_limit = 625

    for c in range(num_classes):
        start_idx = c * samples_per_class
        subset_indices.extend(list(range(start_idx, start_idx + subsample_limit)))
        
    subset_indices = np.array(subset_indices, dtype=np.int32)
    np.save("subset_indices.npy", subset_indices)
    print(f"Stratified indices saved to 'subset_indices.npy' (Total: {len(subset_indices)}).")

    # 2. Instantiate custom dataset and dataloader
    dataset_path = "data/DSRSID.mat"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
        
    print(f"Loading dataset from: {dataset_path}...")
    dataset = DSRSIDDataset(file_path=dataset_path, indices=subset_indices)
    
    # Using batch_size=64, num_workers=0 is safe for h5py on Windows to avoid process-pickling issues
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # 3. Load pretrained ResNet18 model
    print("Initializing pretrained ResNet18 model...")
    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        print("Warning: Failed to load ResNet18 with ResNet18_Weights, falling back to older pretrained=True API.")
        model = resnet18(pretrained=True)

    # Remove the classification layer by replacing it with Identity
    # This turns the model into a 512-dimensional feature extractor
    model.fc = nn.Identity()
    
    # Put model in evaluation mode
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Running inference on device: {device}")

    # 4. Extract embeddings for both modalities
    pan_embeddings_list = []
    mul_embeddings_list = []
    labels_list = []

    print("Extracting features from PAN and MUL modalities...")
    with torch.no_grad():
        for pan_batch, mul_batch, label_batch in tqdm(dataloader, desc="Batches"):
            # Move inputs to device
            pan_batch = pan_batch.to(device)
            mul_batch = mul_batch.to(device)

            # Generate 512-dimensional features
            # ResNet18 expects 3-channel normalized input of shape (batch, 3, 224, 224)
            pan_feats = model(pan_batch)
            mul_feats = model(mul_batch)

            # Store features as numpy arrays
            pan_embeddings_list.append(pan_feats.cpu().numpy())
            mul_embeddings_list.append(mul_feats.cpu().numpy())
            labels_list.append(label_batch.numpy())

    # Concatenate features along the batch dimension
    pan_embeddings = np.concatenate(pan_embeddings_list, axis=0)
    mul_embeddings = np.concatenate(mul_embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # 5. Save the embeddings and class labels
    np.save("pan_embeddings.npy", pan_embeddings)
    np.save("mul_embeddings.npy", mul_embeddings)
    np.save("labels.npy", labels)

    print("\nFeature extraction completed successfully!")
    print(f"Saved 'pan_embeddings.npy' of shape: {pan_embeddings.shape}")
    print(f"Saved 'mul_embeddings.npy' of shape: {mul_embeddings.shape}")
    print(f"Saved 'labels.npy' of shape: {labels.shape}")

    # TODO: Experiment with all 4 spectral channels during advanced training phase
    # (Currently, only the first 3 channels are used as RGB input for feature extraction and visualization).

if __name__ == "__main__":
    main()
