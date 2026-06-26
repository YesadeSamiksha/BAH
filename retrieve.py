import os
import numpy as np
import faiss
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import DSRSIDDataset
from config import (
    DATASET_PATH,
    EMBEDDING_DIR,
    FAISS_DIR,
    OUTPUT_DIR,
    RETRIEVAL_MODE,
    save_versioned_file,
    verify_cache,
    update_pipeline_manifest
)

def retrieve(query_embedding, faiss_index, top_k=5, exclude_index=None):
    """
    Retrieves Top-K nearest neighbors from the FAISS index.
    """
    query_vector = query_embedding.reshape(1, -1).astype('float32')
    search_k = top_k + 1 if exclude_index is not None else top_k
    distances, indices = faiss_index.search(query_vector, search_k)
    
    retrieved_indices = []
    retrieved_distances = []
    
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        if exclude_index is not None and idx == exclude_index:
            continue
        retrieved_indices.append(int(idx))
        retrieved_distances.append(float(dist))
        if len(retrieved_indices) == top_k:
            break
            
    return retrieved_indices[:top_k], retrieved_distances[:top_k]

def evaluate_mode_vectorized(query_embeddings, labels, faiss_index, mode_name, top_k=5):
    """
    Evaluates retrieval performance across all database samples using vectorized ops.
    """
    query_embeddings = query_embeddings.astype('float32')
    num_queries = len(query_embeddings)
    
    # Batch search
    distances, indices = faiss_index.search(query_embeddings, top_k + 1)
    
    # Exclude self-match
    filtered_indices = np.zeros((num_queries, top_k), dtype=np.int64)
    query_indices = np.arange(num_queries).reshape(-1, 1)
    is_self = (indices == query_indices)
    for i in range(num_queries):
        row_idx = indices[i]
        row_self = is_self[i]
        non_self = row_idx[~row_self]
        filtered_indices[i] = non_self[:top_k]
        
    retrieved_labels = labels[filtered_indices]
    query_labels = labels.reshape(-1, 1)
    
    # Compute metrics
    rel = (retrieved_labels == query_labels).astype(np.float32)
    precisions = np.sum(rel, axis=1) / top_k
    
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    label_to_count = dict(zip(unique_labels, class_counts))
    total_relevant = np.array([label_to_count[l] for l in labels], dtype=np.float32) - 1.0
    total_relevant = np.maximum(total_relevant, 1.0)
    recalls = np.sum(rel, axis=1) / total_relevant
    
    sum_pr = precisions + recalls
    f1s = np.zeros_like(precisions)
    non_zero_mask = sum_pr > 0
    f1s[non_zero_mask] = (2 * precisions[non_zero_mask] * recalls[non_zero_mask]) / sum_pr[non_zero_mask]
    
    avg_p = np.mean(precisions)
    avg_r = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    
    print(f"\n[{mode_name}] Average Metrics @ {top_k}:")
    print(f"  Precision: {avg_p:.5f}")
    print(f"  Recall:    {avg_r:.5f}")
    print(f"  F1-Score:  {avg_f1:.5f}")
    
    return avg_p, avg_r, avg_f1

def visualize_query(dataset, query_idx, pan_embeddings, mul_index, labels, output_path=None):
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "retrieval_results.png")
    print(f"\nGenerating visualization for query index {query_idx}...")
    
    query_emb = pan_embeddings[query_idx]
    query_label = labels[query_idx]
    
    retrieved_idxs, retrieved_dists = retrieve(query_emb, mul_index, top_k=5, exclude_index=query_idx)
    query_pan_pil, query_mul_pil, _ = dataset.get_visualization_images(query_idx)
    
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 7, 1)
    plt.imshow(query_pan_pil)
    plt.title(f"Query PAN\nClass: {query_label:.0f}", fontsize=10, weight='bold')
    plt.axis("off")
    
    plt.subplot(1, 7, 2)
    plt.imshow(query_mul_pil)
    plt.title(f"Paired MUL\nClass: {query_label:.0f}", fontsize=10, color='gray')
    plt.axis("off")
    
    for r, (ret_idx, dist) in enumerate(zip(retrieved_idxs, retrieved_dists)):
        _, ret_mul_pil, ret_label = dataset.get_visualization_images(ret_idx)
        is_correct = (ret_label == query_label)
        color = 'green' if is_correct else 'red'
        match_text = "Match" if is_correct else "Mismatch"
        
        plt.subplot(1, 7, r + 3)
        plt.imshow(ret_mul_pil)
        plt.title(
            f"Rank {r+1} (Class {ret_label:.0f})\nDist: {dist:.3f}\n[{match_text}]",
            fontsize=9,
            color=color,
            weight='bold' if is_correct else 'normal'
        )
        plt.axis("off")
        
    plt.suptitle(
        f"Cross-Modal Retrieval Demonstration (PAN -> MUL) | Query Index {query_idx}", 
        fontsize=14, 
        weight='bold', 
        y=0.98
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    save_versioned_file(output_path)
    print(f"Visualization saved to '{output_path}'.")

def main():
    if RETRIEVAL_MODE == "baseline":
        print("Running retrieval in BASELINE mode...")
        pan_index_file = os.path.join(FAISS_DIR, "pan_index.bin")
        mul_index_file = os.path.join(FAISS_DIR, "mul_index.bin")
        pan_embeddings_file = os.path.join(EMBEDDING_DIR, "pan_embeddings.npy")
        mul_embeddings_file = os.path.join(EMBEDDING_DIR, "mul_embeddings.npy")
    elif RETRIEVAL_MODE == "contrastive":
        print("Running retrieval in CONTRASTIVE mode...")
        pan_index_file = os.path.join(FAISS_DIR, "pan_index_contrastive.bin")
        mul_index_file = os.path.join(FAISS_DIR, "mul_index_contrastive.bin")
        pan_embeddings_file = os.path.join(EMBEDDING_DIR, "pan_embeddings_contrastive.npy")
        mul_embeddings_file = os.path.join(EMBEDDING_DIR, "mul_embeddings_contrastive.npy")
    else:
        raise ValueError(f"Unknown RETRIEVAL_MODE: {RETRIEVAL_MODE}")

    labels_file = os.path.join(EMBEDDING_DIR, "labels.npy")
    indices_file = os.path.join(EMBEDDING_DIR, "subset_indices.npy")
    output_path = os.path.join(OUTPUT_DIR, "retrieval_results.png")
    
    # Skip Check
    print("Checking Retrieval...")
    is_cached, reason = verify_cache("retrieval_complete")
    if is_cached:
        print("✔ Compatible cache found.\n")
        return
    else:
        if reason:
            print(f"⚠ Cache invalid. Reason: {reason}")
        print("Running cross-modal retrieval...")

    # Check if necessary files exist
    required_files = [pan_index_file, mul_index_file, pan_embeddings_file, mul_embeddings_file, labels_file, indices_file]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file '{file}' not found. Please build features and indices first.")
            
    # Load indices and dataset
    subset_indices = np.load(indices_file)
    dataset = DSRSIDDataset(file_path=DATASET_PATH, indices=subset_indices)
    
    # Load embeddings with memmap
    pan_embeddings = np.load(pan_embeddings_file, mmap_mode='r')
    mul_embeddings = np.load(mul_embeddings_file, mmap_mode='r')
    labels = np.load(labels_file)
    
    # Load FAISS indices
    print("Loading FAISS indices...")
    pan_index = faiss.read_index(pan_index_file)
    mul_index = faiss.read_index(mul_index_file)
    
    print("\n--- Evaluating Retrieval Modes (Vectorized) ---")
    
    # Mode 1: PAN -> PAN
    evaluate_mode_vectorized(pan_embeddings, labels, pan_index, "PAN -> PAN")
    
    # Mode 2: MUL -> MUL
    evaluate_mode_vectorized(mul_embeddings, labels, mul_index, "MUL -> MUL")
    
    # Mode 3: PAN -> MUL (Cross-Modal)
    evaluate_mode_vectorized(pan_embeddings, labels, mul_index, "PAN -> MUL")
    
    # Mode 4: MUL -> PAN (Cross-Modal)
    evaluate_mode_vectorized(mul_embeddings, labels, pan_index, "MUL -> PAN")
    
    # Visualize a sample query
    sample_query_idx = min(1500, len(pan_embeddings) - 1)
    visualize_query(dataset, sample_query_idx, pan_embeddings, mul_index, labels, output_path)
    
    dataset.close()
    
    # Save stage state
    update_pipeline_manifest("retrieval_complete", True)

if __name__ == "__main__":
    main()
