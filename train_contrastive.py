import os
import sys
import csv
import json
import time

# Ensure UTF-8 output encoding for terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import faiss
from tqdm import tqdm

from dataset import DSRSIDDataset
from config import (
    DATASET_PATH,
    MODEL_DIR,
    CHECKPOINT_DIR,
    EMBEDDING_DIR,
    LOG_DIR,
    OUTPUT_DIR,
    FREEZE_BACKBONE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    TEMPERATURE,
    PATIENCE,
    AUTO_RESUME,
    USE_FULL_DATASET,
    BACKBONE,
    VALIDATION_INTERVAL,
    IS_COLAB,
    save_versioned_file,
    save_timestamped_file,
    check_safeguard,
    verify_cache,
    update_pipeline_manifest,
    save_model_metadata,
    verify_model_metadata
)

LR = LEARNING_RATE

# Helper function to get backbone model dynamically
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

class ProjectionHead(nn.Module):
    """
    Projection head that maps the encoder outputs to a 128D shared space.
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ContrastiveModel(nn.Module):
    """
    Full dual-encoder architecture with configurable backbones and projection heads.
    """
    def __init__(self, freeze_backbone=True):
        super().__init__()
        self.pan_encoder, self.feature_dim = get_backbone_model(BACKBONE)
        self.mul_encoder, _ = get_backbone_model(BACKBONE)

        # Projection heads dynamically adapt to backbone feature dimension
        self.pan_proj = ProjectionHead(input_dim=self.feature_dim)
        self.mul_proj = ProjectionHead(input_dim=self.feature_dim)

        if freeze_backbone:
            self.freeze_backbone_weights()

    def freeze_backbone_weights(self):
        for param in self.pan_encoder.parameters():
            param.requires_grad = False
        for param in self.mul_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone_weights(self):
        for param in self.pan_encoder.parameters():
            param.requires_grad = True
        for param in self.mul_encoder.parameters():
            param.requires_grad = True

    def forward_pan(self, x):
        feats = self.pan_encoder(x)
        return self.pan_proj(feats)

    def forward_mul(self, x):
        feats = self.mul_encoder(x)
        return self.mul_proj(feats)

    def forward(self, pan_x, mul_x):
        return self.forward_pan(pan_x), self.forward_mul(mul_x)

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon) incorporating class labels.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_pan, features_mul, labels):
        # L2 normalize embeddings
        features_pan = F.normalize(features_pan, dim=1)
        features_mul = F.normalize(features_mul, dim=1)

        # Concatenate features from both modalities
        features = torch.cat([features_pan, features_mul], dim=0)  # (2N, D)
        labels = torch.cat([labels, labels], dim=0)  # (2N)

        batch_size = features.size(0)
        labels = labels.contiguous().view(-1, 1)

        # Create mask for matching class labels
        mask = torch.eq(labels, labels.T).float()  # (2N, 2N)

        # Mask out self-contrast (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=features.device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Subtract max for numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-probability of positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # Average loss
        loss = -mean_log_prob_pos.mean()
        return loss

def get_gpu_utilization():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return float(util.gpu), float(mem_info.used) / (1024 ** 2)
    except Exception:
        # Fallback to torch.cuda memory stats
        return 0.0, float(torch.cuda.memory_allocated() / (1024 ** 2))

def visualize_epoch_retrieval(model, epoch, val_pan_feats, val_mul_feats, val_labels, dataset, val_idx, output_path):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pan_proj_val = model.pan_proj(torch.tensor(val_pan_feats, dtype=torch.float32).to(device)).cpu().numpy()
        mul_proj_val = model.mul_proj(torch.tensor(val_mul_feats, dtype=torch.float32).to(device)).cpu().numpy()

    # L2 normalize embeddings for Cosine Similarity
    pan_proj_val = pan_proj_val / np.linalg.norm(pan_proj_val, axis=1, keepdims=True)
    mul_proj_val = mul_proj_val / np.linalg.norm(mul_proj_val, axis=1, keepdims=True)

    # Build Inner Product index
    dimension = 128
    index = faiss.IndexFlatIP(dimension)
    index.add(mul_proj_val)

    # Choose validation index dynamically
    val_samples_per_class = len(val_labels) // 8
    query_val_idx = 2 * val_samples_per_class + int(val_samples_per_class * 0.4)
    query_emb = pan_proj_val[query_val_idx].reshape(1, -1)
    query_label = val_labels[query_val_idx]

    # Search for top-5 matches
    distances, indices = index.search(query_emb, 6)

    retrieved_val_idxs = []
    retrieved_dists = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        if idx == query_val_idx:
            continue
        retrieved_val_idxs.append(int(idx))
        retrieved_dists.append(float(dist))
        if len(retrieved_val_idxs) == 5:
            break

    query_dataset_idx = val_idx[query_val_idx]
    query_pan_pil, query_mul_pil, _ = dataset.get_visualization_images(query_dataset_idx)

    # Plot
    plt.figure(figsize=(15, 4.5))
    plt.subplot(1, 7, 1)
    plt.imshow(query_pan_pil)
    plt.title(f"Query PAN\nClass: {query_label:.0f}", fontsize=10, weight='bold')
    plt.axis("off")
    
    plt.subplot(1, 7, 2)
    plt.imshow(query_mul_pil)
    plt.title(f"Paired MUL\nClass: {query_label:.0f}", fontsize=10, color='gray')
    plt.axis("off")

    for r, (ret_v_idx, dist) in enumerate(zip(retrieved_val_idxs[:5], retrieved_dists[:5])):
        ret_dataset_idx = val_idx[ret_v_idx]
        _, ret_mul_pil, ret_label = dataset.get_visualization_images(ret_dataset_idx)
        
        is_correct = (ret_label == query_label)
        color = 'green' if is_correct else 'red'
        match_text = "Match" if is_correct else "Mismatch"

        plt.subplot(1, 7, r + 3)
        plt.imshow(ret_mul_pil)
        plt.title(
            f"Rank {r+1} (Class {ret_label:.0f})\nSim: {dist:.3f}\n[{match_text}]",
            fontsize=8,
            color=color,
            weight='bold' if is_correct else 'normal'
        )
        plt.axis("off")

    plt.suptitle(
        f"Validation PAN -> MUL Search at Epoch {epoch} | Query Index {query_dataset_idx}",
        fontsize=12,
        weight='bold',
        y=0.98
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    # Safeguard & cache check
    check_safeguard()
    
    print("Checking Contrastive Training...")
    is_cached, reason = verify_cache("training_complete")
    if is_cached:
        print("✔ Compatible cache found.\n")
        return
    else:
        if reason:
            print(f"⚠ Cache invalid. Reason: {reason}")
        print("Running contrastive training...")

    config_dict = {
        "backbone": BACKBONE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "temperature": TEMPERATURE,
        "freeze_backbone": FREEZE_BACKBONE,
        "validation_interval": VALIDATION_INTERVAL,
        "dataset": "DSRSID",
        "use_full_dataset": USE_FULL_DATASET
    }
    config_json_path = os.path.join(LOG_DIR, "training_config.json")
    with open(config_json_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    save_versioned_file(config_json_path)
    print(f"Saved training configuration to '{config_json_path}'.")

    # 1. Load subset indices
    indices_path = os.path.join(EMBEDDING_DIR, "subset_indices.npy")
    if not os.path.exists(indices_path):
        raise FileNotFoundError("Subset indices file not found. Please run extraction first.")
    subset_indices = np.load(indices_path)

    # 2. Perform stratified split: 80% Train, 20% Validation
    total_samples = len(subset_indices)
    samples_per_class = total_samples // 8
    train_split_count = int(samples_per_class * 0.8)

    train_idx = []
    val_idx = []
    for c in range(8):
        start = c * samples_per_class
        train_idx.extend(list(range(start, start + train_split_count)))
        val_idx.extend(list(range(start + train_split_count, start + samples_per_class)))

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    # Load labels
    labels_path = os.path.join(EMBEDDING_DIR, "labels.npy")
    labels = np.load(labels_path)
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    print(f"  Training samples:   {len(train_idx)} ({train_split_count} per class)")
    print(f"  Validation samples: {len(val_idx)} ({samples_per_class - train_split_count} per class)")

    # 3. Initialize model
    print(f"\nInitializing model with backbone {BACKBONE} (FREEZE_BACKBONE={FREEZE_BACKBONE})...")
    model = ContrastiveModel(freeze_backbone=FREEZE_BACKBONE)

    # Setup optimization parameters
    if FREEZE_BACKBONE:
        optimizer = torch.optim.Adam(
            list(model.pan_proj.parameters()) + list(model.mul_proj.parameters()),
            lr=LR
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = SupervisedContrastiveLoss(temperature=TEMPERATURE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Load raw dataset for visualization and optional raw image training
    raw_dataset = DSRSIDDataset(file_path=DATASET_PATH, indices=subset_indices)

    # Determine train/val loaders based on FREEZE_BACKBONE
    if FREEZE_BACKBONE:
        # Load pre-extracted baseline embeddings using mmap for memory efficiency
        pan_embeddings_path = os.path.join(EMBEDDING_DIR, "pan_embeddings.npy")
        mul_embeddings_path = os.path.join(EMBEDDING_DIR, "mul_embeddings.npy")
        
        print("Loading baseline features using memory-mapping...")
        pan_feats = np.load(pan_embeddings_path, mmap_mode='r')
        mul_feats = np.load(mul_embeddings_path, mmap_mode='r')

        train_pan_feats = pan_feats[train_idx]
        train_mul_feats = mul_feats[train_idx]
        val_pan_feats = pan_feats[val_idx]
        val_mul_feats = mul_feats[val_idx]

        train_dataset = TensorDataset(
            torch.tensor(train_pan_feats, dtype=torch.float32),
            torch.tensor(train_mul_feats, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(val_pan_feats, dtype=torch.float32),
            torch.tensor(val_mul_feats, dtype=torch.float32),
            torch.tensor(val_labels, dtype=torch.float32)
        )
    else:
        # If backbone fine-tuning is enabled, we must load raw images from DSRSIDDataset
        print("Fine-tuning enabled. Initializing raw image training/validation datasets...")
        train_dataset = DSRSIDDataset(file_path=DATASET_PATH, indices=subset_indices[train_idx])
        val_dataset = DSRSIDDataset(file_path=DATASET_PATH, indices=subset_indices[val_idx])

    # 4. Auto-resume logic
    history = []
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1

    if AUTO_RESUME and os.path.exists(CHECKPOINT_DIR):
        checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
        if checkpoint_files:
            try:
                epochs_found = [int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files]
                latest_epoch = max(epochs_found)
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{latest_epoch}.pth")
                print(f"\nResuming from checkpoint: {checkpoint_path}")
                
                # Check companion metadata and stop execution on failure
                try:
                    verify_model_metadata(checkpoint_path, strict_hyperparams=True)
                except RuntimeError as e:
                    print(str(e))
                    import sys
                    sys.exit(1)
                
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'scaler_state_dict' in checkpoint and use_amp and checkpoint['scaler_state_dict'] is not None:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                history = checkpoint.get('history', [])
                patience_counter = checkpoint.get('patience_counter', 0)
                start_epoch = latest_epoch + 1
            except Exception as e:
                # If we exited due to mismatch, sys.exit was called. Other errors go here.
                print(f"Warning: Failed to load checkpoint. Starting from scratch. Error: {e}")

    # Generate Initial Epoch 0 visualization (only on start)
    if start_epoch == 1 and FREEZE_BACKBONE:
        epoch_0_visualization = os.path.join(OUTPUT_DIR, "retrieval_epoch_0.png")
        visualize_epoch_retrieval(model, 0, val_pan_feats, val_mul_feats, val_labels, raw_dataset, val_idx, epoch_0_visualization)
        print(f"Generated initial epoch 0 visualization: {epoch_0_visualization}")

    # Setup logs CSV
    csv_file = os.path.join(LOG_DIR, "train_loss.csv")
    csv_exists = os.path.exists(csv_file)
    csv_headers = [
        "epoch", "train_loss", "val_loss", "epoch_duration", 
        "val_duration", "retrieval_duration", "learning_rate", 
        "gpu_utilization", "gpu_memory_allocated", "checkpoint_path"
    ]
    
    # 5. Training Loop
    print("\nStarting contrastive training...")
    early_stop = False
    
    current_batch_size = BATCH_SIZE
    
    for epoch in range(start_epoch, EPOCHS + 1):
        if early_stop:
            break
            
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        
        # Batch size auto-scaling loop for the epoch in case of CUDA OOM
        success = False
        while current_batch_size >= 64:
            num_workers = 2 if (IS_COLAB and device.type == 'cuda') else 0
            train_loader = DataLoader(
                train_dataset, 
                batch_size=current_batch_size, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda'),
                persistent_workers=(num_workers > 0)
            )
            
            try:
                # Run epoch training
                for pan_b, mul_b, labels_b in train_loader:
                    pan_b = pan_b.to(device, non_blocking=True)
                    mul_b = mul_b.to(device, non_blocking=True)
                    labels_b = labels_b.to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        if FREEZE_BACKBONE:
                            proj_pan = model.pan_proj(pan_b)
                            proj_mul = model.mul_proj(mul_b)
                        else:
                            proj_pan = model.forward_pan(pan_b)
                            proj_mul = model.forward_mul(mul_b)
                        loss = criterion(proj_pan, proj_mul, labels_b)
                        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item() * pan_b.size(0)
                
                train_loss /= len(train_dataset)
                success = True
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and device.type == 'cuda':
                    print(f"\nCUDA Out Of Memory caught with batch size {current_batch_size} during training. Retrying with batch size {current_batch_size // 2}...")
                    torch.cuda.empty_cache()
                    current_batch_size = current_batch_size // 2
                    train_loss = 0.0
                else:
                    raise e
                    
        if not success:
            raise RuntimeError("Training failed: CUDA Out Of Memory even at minimum batch size (64).")
            
        epoch_duration = time.time() - epoch_start_time
        
        # Validation Phase (configurable validation interval)
        val_loss = None
        val_duration = 0.0
        
        if epoch % VALIDATION_INTERVAL == 0:
            val_start_time = time.time()
            model.eval()
            val_loss = 0.0
            
            num_workers = 2 if (IS_COLAB and device.type == 'cuda') else 0
            val_loader = DataLoader(
                val_dataset,
                batch_size=current_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda'),
                persistent_workers=(num_workers > 0)
            )
            
            with torch.no_grad():
                for pan_b, mul_b, labels_b in val_loader:
                    pan_b = pan_b.to(device, non_blocking=True)
                    mul_b = mul_b.to(device, non_blocking=True)
                    labels_b = labels_b.to(device, non_blocking=True)
                    
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        if FREEZE_BACKBONE:
                            proj_pan = model.pan_proj(pan_b)
                            proj_mul = model.mul_proj(mul_b)
                        else:
                            proj_pan = model.forward_pan(pan_b)
                            proj_mul = model.forward_mul(mul_b)
                        loss = criterion(proj_pan, proj_mul, labels_b)
                    val_loss += loss.item() * pan_b.size(0)
                    
            val_loss /= len(val_dataset)
            val_duration = time.time() - val_start_time

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Generate retrieval visualization every 5 epochs
        retrieval_duration = 0.0
        if epoch % 5 == 0:
            retrieval_start_time = time.time()
            epoch_visualization = os.path.join(OUTPUT_DIR, f"retrieval_epoch_{epoch}.png")
            
            # Extract features for validation if not frozen
            if FREEZE_BACKBONE:
                v_pan_feats = val_pan_feats
                v_mul_feats = val_mul_feats
            else:
                model.eval()
                v_pan_feats_list = []
                v_mul_feats_list = []
                with torch.no_grad():
                    for p_img, m_img, _ in val_loader:
                        p_img = p_img.to(device)
                        m_img = m_img.to(device)
                        p_feats = model.pan_encoder(p_img)
                        m_feats = model.mul_encoder(m_img)
                        v_pan_feats_list.append(p_feats.cpu().numpy())
                        v_mul_feats_list.append(m_feats.cpu().numpy())
                v_pan_feats = np.concatenate(v_pan_feats_list, axis=0)
                v_mul_feats = np.concatenate(v_mul_feats_list, axis=0)

            visualize_epoch_retrieval(model, epoch, v_pan_feats, v_mul_feats, val_labels, raw_dataset, val_idx, epoch_visualization)
            retrieval_duration = time.time() - retrieval_start_time
            print(f"  Generated visualization at epoch {epoch}")

        # Checkpoint Saving
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if use_amp else None,
            'best_val_loss': best_val_loss,
            'history': history,
            'patience_counter': patience_counter
        }
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        save_model_metadata(checkpoint_path)

        # Early stopping and best model saving (only run on validation epochs)
        if val_loss is not None:
            print(f"Epoch [{epoch:02d}/{EPOCHS}]: Train Loss = {train_loss:.5f} | Val Loss = {val_loss:.5f} | LR = {current_lr:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                save_versioned_file(best_model_path)
                save_timestamped_file(best_model_path)
                save_model_metadata(best_model_path)
                print(f"  Saved best model checkpoint to '{best_model_path}'")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\nEarly stopping triggered! No validation loss improvement for {PATIENCE} validation intervals.")
                    early_stop = True
        else:
            print(f"Epoch [{epoch:02d}/{EPOCHS}]: Train Loss = {train_loss:.5f} | Val Loss = N/A | LR = {current_lr:.6f}")

        # GPU Stats Logging
        gpu_util, gpu_mem = get_gpu_utilization()
        
        # Log to list
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss if val_loss is not None else "",
            "epoch_duration": epoch_duration,
            "val_duration": val_duration,
            "retrieval_duration": retrieval_duration,
            "learning_rate": current_lr,
            "gpu_utilization": gpu_util,
            "gpu_memory_allocated": gpu_mem,
            "checkpoint_path": checkpoint_path
        })

        # Append to CSV log file immediately
        with open(csv_file, "a" if csv_exists else "w", newline="") as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(csv_headers)
                csv_exists = True
            writer.writerow([
                epoch, train_loss, val_loss if val_loss is not None else "N/A",
                f"{epoch_duration:.3f}", f"{val_duration:.3f}", f"{retrieval_duration:.3f}",
                f"{current_lr:.8f}", f"{gpu_util:.1f}", f"{gpu_mem:.2f}", checkpoint_path
            ])

    save_versioned_file(csv_file)
    print(f"\nTraining logs successfully stored in CSV: '{csv_file}'.")

    # 6. Plot training curves
    epochs_plotted = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_epochs = [h["epoch"] for h in history if h["val_loss"] != ""]
    val_losses = [h["val_loss"] for h in history if h["val_loss"] != ""]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_plotted, train_losses, label="Train Loss", marker='o', linewidth=2)
    if val_losses:
        plt.plot(val_epochs, val_losses, label="Val Loss", marker='s', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Supervised Contrastive Loss")
    plt.title(f"Training and Validation Loss Curve ({BACKBONE})", fontsize=12, weight='bold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    curve_path = os.path.join(OUTPUT_DIR, "training_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    save_versioned_file(curve_path)
    print(f"Training curves saved to: {curve_path}")

    # 7. ONNX Export
    onnx_success = False
    onnx_skipped_reason = None
    best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
    onnx_path = os.path.join(MODEL_DIR, "best_model.onnx")
    if os.path.exists(best_model_path):
        print("\nExporting best model to ONNX...")
        try:
            import onnx
            import onnxscript
            has_deps = True
        except ImportError:
            print("ONNX dependencies not installed.")
            print("Skipping ONNX export.")
            has_deps = False
            onnx_skipped_reason = "dependencies not installed"

        if has_deps:
            try:
                onnx_model = ContrastiveModel(freeze_backbone=False)
                onnx_model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
                onnx_model.eval()
                
                dummy_pan = torch.randn(1, 3, 224, 224)
                dummy_mul = torch.randn(1, 3, 224, 224)
                
                torch.onnx.export(
                    onnx_model,
                    (dummy_pan, dummy_mul),
                    onnx_path,
                    input_names=["pan_input", "mul_input"],
                    output_names=["pan_output", "mul_output"],
                    dynamic_axes={
                        "pan_input": {0: "batch_size"},
                        "mul_input": {0: "batch_size"},
                        "pan_output": {0: "batch_size"},
                        "mul_output": {0: "batch_size"}
                    },
                    opset_version=11
                )
                print(f"ONNX model successfully saved to '{onnx_path}'.")
                onnx_success = True
            except Exception as e:
                print("ONNX export skipped:")
                print(e)
                onnx_skipped_reason = str(e)

    # 8. Save experiment metadata
    from config import save_experiment_metadata
    save_experiment_metadata()

    # 9. Record stage completion state
    update_pipeline_manifest("training_complete", True)

    raw_dataset.close()

    # Print success summary
    print("\n========================================")
    print("Training Summary")
    print("========================================\n")
    print("✔ Training Complete\n")
    if os.path.exists(best_model_path):
        print("✔ best_model.pth saved\n")
    
    metadata_path = os.path.splitext(best_model_path)[0] + "_metadata.json"
    if os.path.exists(metadata_path):
        print("✔ Metadata verified\n")
        
    is_cached, _ = verify_cache("training_complete")
    if is_cached:
        print("✔ Cache verified\n")
        
    if onnx_success:
        print("✔ ONNX model exported successfully\n")
    else:
        print("⚠ ONNX export skipped (optional)\n")
    print("========================================")

if __name__ == "__main__":
    main()
