import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from tqdm import tqdm

from dataset import DSRSIDDataset
from config import DATASET_PATH, EMBEDDING_DIR, BATCH_SIZE, USE_FULL_DATASET, save_versioned_file

def main():
    # 1. Compute and save stratified indices for reproducibility
    # Dataset has 8 classes, each has exactly 10,000 contiguous samples.
    # We sample the first 625 images from each class (8 * 625 = 5000 samples total).
    # If USE_FULL_DATASET is True, we sample all 10000 images from each class (80000 samples total).
    print("Computing stratified subset indices...")
    subset_indices = []
    num_classes = 8
    samples_per_class = 10000
    subsample_limit = 10000 if USE_FULL_DATASET else 625

    for c in range(num_classes):
        start_idx = c * samples_per_class
        subset_indices.extend(list(range(start_idx, start_idx + subsample_limit)))
        
    subset_indices = np.array(subset_indices, dtype=np.int32)
    indices_path = os.path.join(EMBEDDING_DIR, "subset_indices.npy")
    np.save(indices_path, subset_indices)
    save_versioned_file(indices_path)
    print(f"Stratified indices saved to '{indices_path}' (Total: {len(subset_indices)}).")

    # 2. Instantiate custom dataset and dataloader
    dataset_path = DATASET_PATH
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
        
    print(f"Loading dataset from: {dataset_path}...")
    dataset = DSRSIDDataset(file_path=dataset_path, indices=subset_indices)
    
    # Using dynamic BATCH_SIZE; num_workers=0 is safe for h5py on Windows to avoid process-pickling issues
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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
    pan_embeddings_path = os.path.join(EMBEDDING_DIR, "pan_embeddings.npy")
    mul_embeddings_path = os.path.join(EMBEDDING_DIR, "mul_embeddings.npy")
    labels_path = os.path.join(EMBEDDING_DIR, "labels.npy")

    np.save(pan_embeddings_path, pan_embeddings)
    np.save(mul_embeddings_path, mul_embeddings)
    np.save(labels_path, labels)

    save_versioned_file(pan_embeddings_path)
    save_versioned_file(mul_embeddings_path)
    save_versioned_file(labels_path)

    print("\nFeature extraction completed successfully!")
    print(f"Saved '{pan_embeddings_path}' of shape: {pan_embeddings.shape}")
    print(f"Saved '{mul_embeddings_path}' of shape: {mul_embeddings.shape}")
    print(f"Saved '{labels_path}' of shape: {labels.shape}")

    # TODO: Experiment with all 4 spectral channels during advanced training phase
    # (Currently, only the first 3 channels are used as RGB input for feature extraction and visualization).

if __name__ == "__main__":
    main()
