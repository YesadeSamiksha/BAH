import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from config import EMBEDDING_DIR, OUTPUT_DIR, save_versioned_file

def generate_tsne_plot(pan_embs, mul_embs, labels, title, output_path, samples_per_class=100):
    """
    Fits t-SNE on combined PAN and MUL embeddings and generates a professional scatter plot.
    
    Args:
        pan_embs (np.ndarray): PAN embeddings (N, D).
        mul_embs (np.ndarray): MUL embeddings (N, D).
        labels (np.ndarray): Array of labels corresponding to embeddings (N,).
        title (str): Plot title.
        output_path (str): Filepath to save the plot image.
        samples_per_class (int): Number of samples per class to draw for visualization.
    """
    print(f"Selecting {samples_per_class} samples per class for t-SNE...")
    sampled_indices = []
    for c in range(8):
        class_label = float(c + 1.0)
        class_indices = np.where(labels == class_label)[0]
        sampled_indices.extend(class_indices[:samples_per_class])
    sampled_indices = np.array(sampled_indices)
    
    # Slice embeddings and labels
    pan_sampled = pan_embs[sampled_indices]
    mul_sampled = mul_embs[sampled_indices]
    labels_sampled = labels[sampled_indices]
    
    # Combine PAN and MUL to project them into the same 2D t-SNE space
    # Total samples = 2 * 8 * samples_per_class (e.g. 1600 points)
    combined_embs = np.concatenate([pan_sampled, mul_sampled], axis=0)
    
    print("Fitting t-SNE (this might take a few seconds on CPU)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca')
    embeddings_2d = tsne.fit_transform(combined_embs)
    
    # Split back into PAN and MUL
    num_sampled = len(sampled_indices)
    pan_2d = embeddings_2d[:num_sampled]
    mul_2d = embeddings_2d[num_sampled:]
    
    # Plotting configuration
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    # Class colors (8 distinct colors from qualitative map)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:8]
    class_names = [
        "Class 1 (Label 1.0)",
        "Class 2 (Label 2.0)",
        "Class 3 (Label 3.0)",
        "Class 4 (Label 4.0)",
        "Class 5 (Label 5.0)",
        "Class 6 (Label 6.0)",
        "Class 7 (Label 7.0)",
        "Class 8 (Label 8.0)",
    ]
    
    # Plot classes individually to compile class legends
    for c in range(8):
        class_label = float(c + 1.0)
        idx_match = np.where(labels_sampled == class_label)[0]
        
        # Plot PAN (circle marker 'o', filled)
        plt.scatter(
            pan_2d[idx_match, 0], pan_2d[idx_match, 1],
            color=colors[c], marker='o', s=30, alpha=0.8,
            label=class_names[c] if c == 0 else "" # Class colors are covered by a single class legend
        )
        
        # Plot MUL (triangle-up marker '^', hollow/filled)
        plt.scatter(
            mul_2d[idx_match, 0], mul_2d[idx_match, 1],
            color=colors[c], marker='^', s=45, alpha=0.8,
            edgecolors='black', linewidths=0.5
        )
        
    # Create two separate legends: one for class colors, one for markers (modalities)
    # 1. Modality markers legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=8, label='PAN Modality'),
        Line2D([0], [0], marker='^', color='gray', markerfacecolor='gray', markeredgecolor='black', linestyle='None', markersize=8, label='MUL Modality')
    ]
    modality_legend = plt.legend(handles=legend_elements, loc='upper left', framealpha=0.9, title="Modalities")
    plt.gca().add_artist(modality_legend)
    
    # 2. Class colors legend
    color_elements = [
        Line2D([0], [0], marker='s', color='white', markerfacecolor=colors[c], markersize=10, label=class_names[c])
        for c in range(8)
    ]
    plt.legend(handles=color_elements, loc='upper right', framealpha=0.9, title="Land Cover Classes", ncol=2)
    
    plt.title(title, fontsize=14, weight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("t-SNE Dimension 1", fontsize=10)
    plt.ylabel("t-SNE Dimension 2", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"t-SNE visualization saved to '{output_path}'.")

def main():
    # 1. Load labels
    labels_file = os.path.join(EMBEDDING_DIR, "labels.npy")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file '{labels_file}' not found. Please run baseline extraction first.")
    labels = np.load(labels_file)

    # 2. Generate Before Training Plot (Baseline 512D)
    print("\n--- Processing Baseline (Before Training) Embeddings ---")
    pan_before_file = os.path.join(EMBEDDING_DIR, "pan_embeddings.npy")
    mul_before_file = os.path.join(EMBEDDING_DIR, "mul_embeddings.npy")
    
    if os.path.exists(pan_before_file) and os.path.exists(mul_before_file):
        pan_before = np.load(pan_before_file)
        mul_before = np.load(mul_before_file)
        
        before_path = os.path.join(OUTPUT_DIR, "embeddings_tsne_before.png")
        generate_tsne_plot(
            pan_before, mul_before, labels,
            title="t-SNE Embeddings Space Before Training (Baseline 512D)",
            output_path=before_path
        )
        save_versioned_file(before_path)
    else:
        print("Warning: Baseline embeddings not found. Skipping 'before' plot.")

    # 3. Generate After Training Plot (Contrastive 128D)
    print("\n--- Processing Contrastive (After Training) Embeddings ---")
    pan_after_file = os.path.join(EMBEDDING_DIR, "pan_embeddings_contrastive.npy")
    mul_after_file = os.path.join(EMBEDDING_DIR, "mul_embeddings_contrastive.npy")
    
    if os.path.exists(pan_after_file) and os.path.exists(mul_after_file):
        pan_after = np.load(pan_after_file)
        mul_after = np.load(mul_after_file)
        
        after_path = os.path.join(OUTPUT_DIR, "embeddings_tsne_after.png")
        generate_tsne_plot(
            pan_after, mul_after, labels,
            title="t-SNE Embeddings Space After Supervised Contrastive Training (128D)",
            output_path=after_path
        )
        save_versioned_file(after_path)
    else:
        print("Warning: Contrastive embeddings not found. Skipping 'after' plot.")

if __name__ == "__main__":
    main()
