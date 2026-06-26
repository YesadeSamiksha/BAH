import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset import DSRSIDDataset
from config import (
    DATASET_PATH,
    EMBEDDING_DIR,
    BATCH_SIZE,
    USE_FULL_DATASET,
    BACKBONE,
    IS_COLAB,
    save_versioned_file,
    check_stage_skip,
    save_stage_state
)

# Define get_backbone_model here to avoid import dependencies during Phase 1
def get_backbone_model(backbone_name):
    import torchvision.models as models
    backbone_name = backbone_name.lower()
    if backbone_name == "resnet18":
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            backbone = models.resnet18(pretrained=True)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif backbone_name == "resnet50":
        try:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            backbone = models.resnet50(pretrained=True)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}. Only resnet18 and resnet50 are supported.")
    return backbone, feature_dim

def run_extraction(dataloader, model, device, feature_dim):
    pan_embeddings_list = []
    mul_embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for pan_batch, mul_batch, label_batch in tqdm(dataloader, desc="Extracting features"):
            # Move inputs to device with non_blocking=True for async transfer
            pan_batch = pan_batch.to(device, non_blocking=True)
            mul_batch = mul_batch.to(device, non_blocking=True)

            # Extract features (using mixed precision autocast if GPU available)
            use_amp = (device.type == 'cuda')
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pan_feats = model(pan_batch)
                mul_feats = model(mul_batch)

            if len(pan_embeddings_list) == 0:
                print(f"Batch Output      : {pan_feats.shape}")

            # Store features as numpy arrays
            pan_embeddings_list.append(pan_feats.cpu().numpy())
            mul_embeddings_list.append(mul_feats.cpu().numpy())
            labels_list.append(label_batch.numpy())

    pan_embs = np.concatenate(pan_embeddings_list, axis=0)
    mul_embs = np.concatenate(mul_embeddings_list, axis=0)
    lbls = np.concatenate(labels_list, axis=0)
    print(f"Saved Shape       : {pan_embs.shape}")
    return pan_embs, mul_embs, lbls

def main():
    # CPU Safeguard
    from config import check_safeguard, verify_cache
    check_safeguard()

    # 1. Compute and save stratified indices for reproducibility
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
    
    # 2. Skip check
    # Check embedding dimension beforehand
    _, test_feature_dim = get_backbone_model(BACKBONE)
    
    pan_embeddings_path = os.path.join(EMBEDDING_DIR, "pan_embeddings.npy")
    mul_embeddings_path = os.path.join(EMBEDDING_DIR, "mul_embeddings.npy")
    labels_path = os.path.join(EMBEDDING_DIR, "labels.npy")
    
    expected_outputs = [indices_path, pan_embeddings_path, mul_embeddings_path, labels_path]
    extra_config = {
        "use_full_dataset": USE_FULL_DATASET,
        "embedding_dim": test_feature_dim
    }
    
    print("Checking Baseline Embeddings...")
    is_cached, reason = verify_cache("baseline_embeddings")
    if is_cached:
        print("✔ Compatible cache found.\n")
        return
    else:
        if reason:
            print(f"⚠ Cache invalid. Reason: {reason}")
        print("Extracting baseline embeddings...")

    np.save(indices_path, subset_indices)
    save_versioned_file(indices_path)
    print(f"Stratified indices saved to '{indices_path}' (Total: {len(subset_indices)}).")

    # 3. Instantiate custom dataset
    dataset_path = DATASET_PATH
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
        
    print(f"Loading dataset from: {dataset_path}...")
    dataset = DSRSIDDataset(file_path=dataset_path, indices=subset_indices)
    
    # 4. Load pretrained backbone model
    print(f"Initializing pretrained {BACKBONE} model...")
    model, feature_dim = get_backbone_model(BACKBONE)
    model.eval()

    # Diagnostic prints
    print(f"Selected Backbone : {BACKBONE}")
    print(f"Model Class       : {model.__class__.__module__}.{model.__class__.__name__}")
    print(f"Feature Dimension : {feature_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Running inference on device: {device}")

    # 5. Extract embeddings with Automatic Batch Size Scaling
    current_batch_size = BATCH_SIZE
    success = False
    
    while current_batch_size >= 64:
        print(f"Running extraction with batch_size={current_batch_size}...")
        
        # Setup data loader optimized for device type
        # pin_memory, num_workers, and persistent_workers are set for high GPU utilization
        num_workers = 2 if (IS_COLAB and device.type == 'cuda') else 0
        dataloader = DataLoader(
            dataset,
            batch_size=current_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(num_workers > 0)
        )
        
        try:
            pan_embeddings, mul_embeddings, labels = run_extraction(dataloader, model, device, feature_dim)
            success = True
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == 'cuda':
                print(f"\nCUDA Out Of Memory caught with batch size {current_batch_size}. Retrying with batch size {current_batch_size // 2}...")
                torch.cuda.empty_cache()
                current_batch_size = current_batch_size // 2
            else:
                # Re-raise if it's not a CUDA OOM
                raise e

    if not success:
        raise RuntimeError("Embedding extraction failed: CUDA Out Of Memory even at minimum batch size (64).")

    # 6. Save the embeddings and class labels
    np.save(pan_embeddings_path, pan_embeddings)
    np.save(mul_embeddings_path, mul_embeddings)
    np.save(labels_path, labels)

    save_versioned_file(pan_embeddings_path)
    save_versioned_file(mul_embeddings_path)
    save_versioned_file(labels_path)

    print("\nFeature extraction completed successfully!")
    print(f"PAN Embeddings Shape : {pan_embeddings.shape}")
    print(f"MUL Embeddings Shape : {mul_embeddings.shape}")
    print(f"Saved '{pan_embeddings_path}' of shape: {pan_embeddings.shape}")
    print(f"Saved '{mul_embeddings_path}' of shape: {mul_embeddings.shape}")
    print(f"Saved '{labels_path}' of shape: {labels.shape}")

    # Save experiment metadata
    from config import save_experiment_metadata
    save_experiment_metadata()

    # 7. Record stage completion state
    save_stage_state("extract_embeddings", expected_outputs, extra_config)

if __name__ == "__main__":
    main()
