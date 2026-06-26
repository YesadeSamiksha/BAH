import os
import numpy as np
import faiss
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import DSRSIDDataset
from config import DATASET_PATH, EMBEDDING_DIR, FAISS_DIR, OUTPUT_DIR, RETRIEVAL_MODE, save_versioned_file

def retrieve(query_embedding, faiss_index, top_k=5, exclude_index=None):
    """
    Retrieves Top-K nearest neighbors from the FAISS index.
    
    Args:
        query_embedding (np.ndarray): 512D query feature vector.
        faiss_index (faiss.Index): FAISS index to search.
        top_k (int): Number of nearest neighbors to retrieve.
        exclude_index (int, optional): Index to exclude from results (leave-one-out).
        
    Returns:
        retrieved_indices (list of int): Database indices of top-K results.
        retrieved_distances (list of float): L2 distances of top-K results.
    """
    # Ensure query embedding is 2D float32
    query_vector = query_embedding.reshape(1, -1).astype('float32')
    
    # Request top_k + 1 in case we need to filter out the query index
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

def evaluate_mode(query_embeddings, labels, faiss_index, mode_name, top_k=5):
    """
    Evaluates retrieval performance across all database samples for a specific mode.
    
    Args:
        query_embeddings (np.ndarray): Array of query embeddings.
        labels (np.ndarray): Array of class labels corresponding to indices.
        faiss_index (faiss.Index): FAISS index of the gallery/database modality.
        mode_name (str): Description of the retrieval mode (e.g. "PAN -> MUL").
        top_k (int): Number of retrieved items to evaluate.
        
    Returns:
        avg_precision (float), avg_recall (float), avg_f1 (float)
    """
    precisions = []
    recalls = []
    f1s = []
    
    num_queries = len(query_embeddings)
    for i in range(num_queries):
        query_emb = query_embeddings[i]
        query_label = labels[i]
        
        # Search the database, excluding the query's corresponding index
        retrieved_idxs, _ = retrieve(query_emb, faiss_index, top_k=top_k, exclude_index=i)
        
        # Denominator for Recall: total relevant items in database (excluding the query instance)
        class_count = np.sum(labels == query_label)
        total_relevant = class_count - 1  # 625 - 1 = 624 for stratified subset
        
        if total_relevant <= 0:
            continue
            
        # Count true positives (items matching the query class)
        num_relevant_retrieved = sum(labels[idx] == query_label for idx in retrieved_idxs)
        
        p = num_relevant_retrieved / top_k
        r = num_relevant_retrieved / total_relevant
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        
    avg_p = np.mean(precisions)
    avg_r = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    
    print(f"\n[{mode_name}] Average Metrics @ {top_k}:")
    print(f"  Precision: {avg_p:.5f}")
    print(f"  Recall:    {avg_r:.5f}")
    print(f"  F1-Score:  {avg_f1:.5f}")
    
    return avg_p, avg_r, avg_f1

def visualize_query(dataset, query_idx, pan_embeddings, mul_index, labels, output_path=None):
    """
    Visualizes cross-modal PAN -> MUL retrieval for a single query.
    Displays:
        - Query PAN image
        - True Paired MUL image
        - Top-5 retrieved MUL images with labels and distances
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "retrieval_results.png")
    print(f"\nGenerating visualization for query index {query_idx}...")
    
    query_emb = pan_embeddings[query_idx]
    query_label = labels[query_idx]
    
    # Retrieve top-5 from MUL index, excluding the matched pair itself to show retrieval generalizability
    retrieved_idxs, retrieved_dists = retrieve(query_emb, mul_index, top_k=5, exclude_index=query_idx)
    
    # Get images for plotting
    query_pan_pil, query_mul_pil, _ = dataset.get_visualization_images(query_idx)
    
    plt.figure(figsize=(18, 5))
    
    # 1. Plot Query PAN
    plt.subplot(1, 7, 1)
    plt.imshow(query_pan_pil)
    plt.title(f"Query PAN\nClass: {query_label:.0f}", fontsize=10, weight='bold')
    plt.axis("off")
    
    # 2. Plot True Paired MUL (modality comparison)
    plt.subplot(1, 7, 2)
    plt.imshow(query_mul_pil)
    plt.title(f"Paired MUL\nClass: {query_label:.0f}", fontsize=10, color='gray')
    plt.axis("off")
    
    # 3. Plot Top-5 Retrieved MUL
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
    
    # Check if necessary files exist
    required_files = [pan_index_file, mul_index_file, pan_embeddings_file, mul_embeddings_file, labels_file, indices_file]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file '{file}' not found. Please build features and indices first.")
            
    # Load indices and instantiate Dataset
    subset_indices = np.load(indices_file)
    dataset = DSRSIDDataset(file_path=DATASET_PATH, indices=subset_indices)
    
    # Load embeddings and labels
    pan_embeddings = np.load(pan_embeddings_file)
    mul_embeddings = np.load(mul_embeddings_file)
    labels = np.load(labels_file)
    
    # Load FAISS indices
    print("Loading FAISS indices...")
    pan_index = faiss.read_index(pan_index_file)
    mul_index = faiss.read_index(mul_index_file)
    
    print("\n--- Evaluating Retrieval Modes ---")
    
    # Mode 1: PAN -> PAN
    evaluate_mode(pan_embeddings, labels, pan_index, "PAN -> PAN")
    
    # Mode 2: MUL -> MUL
    evaluate_mode(mul_embeddings, labels, mul_index, "MUL -> MUL")
    
    # Mode 3: PAN -> MUL (Cross-Modal)
    evaluate_mode(pan_embeddings, labels, mul_index, "PAN -> MUL")
    
    # Mode 4: MUL -> PAN (Cross-Modal)
    evaluate_mode(mul_embeddings, labels, pan_index, "MUL -> PAN")
    
    # Visualize a sample cross-modal query (Index 1500 belongs to class 3.0 out of 0..7)
    # Using a fixed index in the middle of Class 3.0 (which resides at indices 1250 - 1874 in our subset)
    sample_query_idx = 1500
    visualize_query(dataset, sample_query_idx, pan_embeddings, mul_index, labels)
    
    dataset.close()

if __name__ == "__main__":
    main()
