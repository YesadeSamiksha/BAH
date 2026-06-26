import os
import json
import numpy as np
import faiss
from config import (
    EMBEDDING_DIR,
    FAISS_DIR,
    LOG_DIR,
    FAISS_INDEX,
    save_versioned_file,
    verify_cache,
    update_pipeline_manifest
)

def compute_metrics_for_mode_vectorized(query_embeddings, labels, faiss_index, exclude_self=True):
    """
    Vectorized calculation of Precision@5, Recall@5, F1@5, mAP@5, and mAP@10.
    """
    query_embeddings = query_embeddings.astype('float32')
    num_queries = len(query_embeddings)
    
    # We search for top-10 nearest neighbors
    search_k = 11 if exclude_self else 10
    distances, indices = faiss_index.search(query_embeddings, search_k)
    
    # Exclude self-match
    filtered_indices = np.zeros((num_queries, 10), dtype=np.int64)
    if exclude_self:
        query_indices = np.arange(num_queries).reshape(-1, 1)
        is_self = (indices == query_indices)
        for i in range(num_queries):
            row_idx = indices[i]
            row_self = is_self[i]
            non_self = row_idx[~row_self]
            filtered_indices[i] = non_self[:10]
    else:
        filtered_indices = indices[:, :10]
        
    retrieved_labels = labels[filtered_indices]
    query_labels = labels.reshape(-1, 1)
    
    # Compute matches
    rel = (retrieved_labels == query_labels).astype(np.float32)
    
    # Running precision for mAP calculation
    rel_cumsum = np.cumsum(rel, axis=1)
    ranks = np.arange(1, 11).astype(np.float32)
    precisions_at_ranks = rel_cumsum / ranks
    
    # AP@5 and AP@10
    ap_5 = np.sum(precisions_at_ranks[:, :5] * rel[:, :5], axis=1) / 5.0
    ap_10 = np.sum(precisions_at_ranks * rel, axis=1) / 10.0
    
    # Metrics@5
    rel_5 = rel[:, :5]
    num_rel_5 = np.sum(rel_5, axis=1)
    precisions_5 = num_rel_5 / 5.0
    
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    label_to_count = dict(zip(unique_labels, class_counts))
    total_relevant = np.array([label_to_count[l] for l in labels], dtype=np.float32)
    if exclude_self:
        total_relevant = total_relevant - 1.0
    total_relevant = np.maximum(total_relevant, 1.0)
    
    recalls_5 = num_rel_5 / total_relevant
    
    sum_pr = precisions_5 + recalls_5
    f1s_5 = np.zeros_like(precisions_5)
    non_zero_mask = sum_pr > 0
    f1s_5[non_zero_mask] = (2 * precisions_5[non_zero_mask] * recalls_5[non_zero_mask]) / sum_pr[non_zero_mask]
    
    return {
        "precision_5": float(np.mean(precisions_5)),
        "recall_5": float(np.mean(recalls_5)),
        "f1_5": float(np.mean(f1s_5)),
        "map_5": float(np.mean(ap_5)),
        "map_10": float(np.mean(ap_10))
    }

def main():
    labels_file = os.path.join(EMBEDDING_DIR, "labels.npy")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file '{labels_file}' not found! Run extraction scripts first.")
        
    # 1. Skip Check
    baseline_file = os.path.join(LOG_DIR, "metrics_summary_baseline.json")
    contrastive_file = os.path.join(LOG_DIR, "metrics_summary_contrastive.json")
    
    print("Checking Precomputed Metrics...")
    is_cached, reason = verify_cache("metrics_ready")
    if is_cached:
        print("✔ Compatible cache found.\n")
        return
    else:
        if reason:
            print(f"⚠ Cache invalid. Reason: {reason}")
        print("Pre-computing evaluation metrics for all configurations...")

    # 2. Load Embeddings and Indices
    labels = np.load(labels_file)
    print("Pre-computing evaluation metrics for all configurations...")

    print("Loading Baseline Embeddings...")
    b_pan_embs = np.load(os.path.join(EMBEDDING_DIR, "pan_embeddings.npy"), mmap_mode='r')
    b_mul_embs = np.load(os.path.join(EMBEDDING_DIR, "mul_embeddings.npy"), mmap_mode='r')
    b_pan_index = faiss.read_index(os.path.join(FAISS_DIR, "pan_index.bin"))
    b_mul_index = faiss.read_index(os.path.join(FAISS_DIR, "mul_index.bin"))

    print("Loading Contrastive Embeddings...")
    c_pan_embs = np.load(os.path.join(EMBEDDING_DIR, "pan_embeddings_contrastive.npy"), mmap_mode='r')
    c_mul_embs = np.load(os.path.join(EMBEDDING_DIR, "mul_embeddings_contrastive.npy"), mmap_mode='r')
    c_pan_index = faiss.read_index(os.path.join(FAISS_DIR, "pan_index_contrastive.bin"))
    c_mul_index = faiss.read_index(os.path.join(FAISS_DIR, "mul_index_contrastive.bin"))

    summary = {
        "baseline": {},
        "contrastive": {}
    }

    # 3. Compute Baseline Metrics (L2 metric)
    print("\nComputing Baseline Metrics...")
    summary["baseline"]["PAN_PAN"] = compute_metrics_for_mode_vectorized(b_pan_embs, labels, b_pan_index, exclude_self=True)
    summary["baseline"]["MUL_MUL"] = compute_metrics_for_mode_vectorized(b_mul_embs, labels, b_mul_index, exclude_self=True)
    summary["baseline"]["PAN_MUL"] = compute_metrics_for_mode_vectorized(b_pan_embs, labels, b_mul_index, exclude_self=True)
    summary["baseline"]["MUL_PAN"] = compute_metrics_for_mode_vectorized(b_mul_embs, labels, b_pan_index, exclude_self=True)

    # 4. Compute Contrastive Metrics (Cosine metric)
    print("\nComputing Contrastive Metrics...")
    summary["contrastive"]["PAN_PAN"] = compute_metrics_for_mode_vectorized(c_pan_embs, labels, c_pan_index, exclude_self=True)
    summary["contrastive"]["MUL_MUL"] = compute_metrics_for_mode_vectorized(c_mul_embs, labels, c_mul_index, exclude_self=True)
    summary["contrastive"]["PAN_MUL"] = compute_metrics_for_mode_vectorized(c_pan_embs, labels, c_mul_index, exclude_self=True)
    summary["contrastive"]["MUL_PAN"] = compute_metrics_for_mode_vectorized(c_mul_embs, labels, c_pan_index, exclude_self=True)

    # 5. Save to JSON files
    with open(baseline_file, "w") as f:
        json.dump(summary["baseline"], f, indent=4)
    save_versioned_file(baseline_file)

    with open(contrastive_file, "w") as f:
        json.dump(summary["contrastive"], f, indent=4)
    save_versioned_file(contrastive_file)
        
    print(f"\nSuccessfully pre-computed all metrics and saved to separate files under '{LOG_DIR}'!")
    print(json.dumps(summary, indent=2))

    # 6. Save stage state
    update_pipeline_manifest("metrics_ready", True)

if __name__ == "__main__":
    main()
