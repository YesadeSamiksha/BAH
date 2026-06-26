import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import matplotlib.pyplot as plt
import faiss
from tqdm import tqdm

from dataset import DSRSIDDataset

# Configuration Constants
FREEZE_BACKBONE = True  # Initially frozen. Can be changed to False to unfreeze later.
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
TEMPERATURE = 0.07
PATIENCE = 5

class ProjectionHead(nn.Module):
    """
    Projection head that maps the 512D ResNet18 encoder outputs to a 128D shared space.
    """
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
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
    Full dual-encoder architecture with ResNet18 backbones and projection heads.
    """
    def __init__(self, freeze_backbone=True):
        super().__init__()
        # Initialize ResNet18 backbones
        try:
            self.pan_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.mul_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            self.pan_encoder = resnet18(pretrained=True)
            self.mul_encoder = resnet18(pretrained=True)

        # Replace classification layer with Identity
        self.pan_encoder.fc = nn.Identity()
        self.mul_encoder.fc = nn.Identity()

        # Projection heads
        self.pan_proj = ProjectionHead()
        self.mul_proj = ProjectionHead()

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

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon) incorporating class labels to pull matching categories
    closer and push different classes apart.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_pan, features_mul, labels):
        """
        Args:
            features_pan (Tensor): PAN features of shape (N, D)
            features_mul (Tensor): MUL features of shape (N, D)
            labels (Tensor): Labels of shape (N)
        """
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

def visualize_epoch_retrieval(model, epoch, val_pan_feats, val_mul_feats, val_labels, dataset, val_idx, output_path):
    """
    Generates and saves a PAN -> MUL retrieval plot using validation set embeddings.
    Allows visual tracking of alignment progress across training epochs.
    """
    model.eval()
    with torch.no_grad():
        # Project validation features to 128D shared space
        pan_proj_val = model.pan_proj(torch.tensor(val_pan_feats, dtype=torch.float32)).numpy()
        mul_proj_val = model.mul_proj(torch.tensor(val_mul_feats, dtype=torch.float32)).numpy()

    # L2 normalize embeddings for Cosine Similarity (Inner Product in FAISS)
    pan_proj_val = pan_proj_val / np.linalg.norm(pan_proj_val, axis=1, keepdims=True)
    mul_proj_val = mul_proj_val / np.linalg.norm(mul_proj_val, axis=1, keepdims=True)

    # Build Inner Product index
    dimension = 128
    index = faiss.IndexFlatIP(dimension)
    index.add(mul_proj_val)

    # Choose validation index 300 (belongs to Class 3.0 out of classes 1.0-8.0)
    query_val_idx = 300
    query_emb = pan_proj_val[query_val_idx].reshape(1, -1)
    query_label = val_labels[query_val_idx]

    # Search for top-5 matches, requesting 6 to filter out self-match if it exists
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

    # Map validation index back to the 5000 stratified dataset index
    query_dataset_idx = val_idx[query_val_idx]
    query_pan_pil, query_mul_pil, _ = dataset.get_visualization_images(query_dataset_idx)

    # Create visualization plot
    plt.figure(figsize=(15, 4.5))
    
    # Query PAN
    plt.subplot(1, 7, 1)
    plt.imshow(query_pan_pil)
    plt.title(f"Query PAN\nClass: {query_label:.0f}", fontsize=10, weight='bold')
    plt.axis("off")
    
    # Paired MUL
    plt.subplot(1, 7, 2)
    plt.imshow(query_mul_pil)
    plt.title(f"Paired MUL\nClass: {query_label:.0f}", fontsize=10, color='gray')
    plt.axis("off")

    # Retrieved MUL results
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
    # 1. Load the pre-extracted 512D embeddings and labels from Phase 1
    # This enables extremely fast projection head training on CPU.
    pan_embeddings_path = "pan_embeddings.npy"
    mul_embeddings_path = "mul_embeddings.npy"
    labels_path = "labels.npy"
    indices_path = "subset_indices.npy"

    if not all(os.path.exists(f) for f in [pan_embeddings_path, mul_embeddings_path, labels_path, indices_path]):
        raise FileNotFoundError(
            "Phase 1 embedding files not found! Please run Phase 1 scripts first."
        )

    print("Loading baseline 512D ResNet18 features...")
    pan_feats = np.load(pan_embeddings_path)
    mul_feats = np.load(mul_embeddings_path)
    labels = np.load(labels_path)
    subset_indices = np.load(indices_path)

    # 2. Perform stratified split: 80% Train, 20% Validation
    # There are 8 classes, each with 625 samples.
    # Train: 500 samples/class (4000 total). Val: 125 samples/class (1000 total).
    print("Creating stratified train/validation split (80/20)...")
    train_idx = []
    val_idx = []
    samples_per_class = 625
    train_split_count = 500

    for c in range(8):
        start = c * samples_per_class
        train_idx.extend(list(range(start, start + train_split_count)))
        val_idx.extend(list(range(start + train_split_count, start + samples_per_class)))

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    # Split features and labels
    train_pan_feats = pan_feats[train_idx]
    train_mul_feats = mul_feats[train_idx]
    train_labels = labels[train_idx]

    val_pan_feats = pan_feats[val_idx]
    val_mul_feats = mul_feats[val_idx]
    val_labels = labels[val_idx]

    print(f"  Training samples:   {len(train_idx)} (500 per class)")
    print(f"  Validation samples: {len(val_idx)} (125 per class)")

    # 3. Create datasets and loaders for fast feature-based training
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load raw dataset for visualization
    raw_dataset = DSRSIDDataset(file_path="data/DSRSID.mat", indices=subset_indices)

    # 4. Initialize model, optimizer, and loss function
    # Only training projection heads, keeping backbone frozen
    print("\nInitializing model and projection heads...")
    model = ContrastiveModel(freeze_backbone=FREEZE_BACKBONE)
    
    # We pass only projection head parameters to the optimizer
    optimizer = torch.optim.Adam(
        list(model.pan_proj.parameters()) + list(model.mul_proj.parameters()),
        lr=LR
    )
    
    criterion = SupervisedContrastiveLoss(temperature=TEMPERATURE)

    # Track losses
    history = []
    best_val_loss = float("inf")
    patience_counter = 0

    # 5. Generate Initial Epoch 0 (Untrained state) visualization
    visualize_epoch_retrieval(model, 0, val_pan_feats, val_mul_feats, val_labels, raw_dataset, val_idx, "retrieval_epoch_0.png")
    print("Generated retrieval_epoch_0.png visualization.")

    # 6. Training Loop
    print("\nStarting projection heads training...")
    for epoch in range(1, EPOCHS + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for pan_b, mul_b, labels_b in train_loader:
            optimizer.zero_grad()
            
            # Project the cached ResNet features to 128D
            proj_pan = model.pan_proj(pan_b)
            proj_mul = model.mul_proj(mul_b)
            
            loss = criterion(proj_pan, proj_mul, labels_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * pan_b.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pan_b, mul_b, labels_b in val_loader:
                proj_pan = model.pan_proj(pan_b)
                proj_mul = model.mul_proj(mul_b)
                loss = criterion(proj_pan, proj_mul, labels_b)
                val_loss += loss.item() * pan_b.size(0)

        val_loss /= len(val_loader.dataset)
        history.append((epoch, train_loss, val_loss))

        print(f"Epoch [{epoch:02d}/{EPOCHS}]: Train Loss = {train_loss:.5f} | Val Loss = {val_loss:.5f}")

        # Epoch-based visualizations at 5 and 10
        if epoch == 5:
            visualize_epoch_retrieval(model, 5, val_pan_feats, val_mul_feats, val_labels, raw_dataset, val_idx, "retrieval_epoch_5.png")
            print("  Generated retrieval_epoch_5.png")
        elif epoch == 10:
            visualize_epoch_retrieval(model, 10, val_pan_feats, val_mul_feats, val_labels, raw_dataset, val_idx, "retrieval_epoch_10.png")
            print("  Generated retrieval_epoch_10.png")

        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best weights only
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered! No validation improvement for {PATIENCE} epochs.")
                break

    # Save validation plot for final epoch if it wasn't captured
    if epoch >= 10:
        # If epoch is greater than 10, let's also save the final epoch retrieval visualization as epoch 10 if epoch 10 was skipped
        pass

    # 7. Save training history to CSV
    csv_file = "train_loss.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(history)
    print(f"\nLoss logs saved to '{csv_file}'.")

    # 8. Plot training curve
    epochs_plotted = [h[0] for h in history]
    train_losses = [h[1] for h in history]
    val_losses = [h[2] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_plotted, train_losses, label="Train Loss", marker='o', linewidth=2)
    plt.plot(epochs_plotted, val_losses, label="Val Loss", marker='s', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Supervised Contrastive Loss")
    plt.title("Training and Validation Loss Curve", fontsize=12, weight='bold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.close()
    print("Training loss curve saved to 'training_curve.png'.")

    raw_dataset.close()

if __name__ == "__main__":
    main()
