import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from config import EMBEDDING_DIR, FAISS_DIR, LOG_DIR, save_versioned_file

def retrieve_ip(query_embedding, faiss_index, top_k=10, exclude_index=None):
    """
    Retrieves Top-K nearest neighbors from the FAISS IndexFlatIP (Cosine Similarity) index.
    """
    query_vector = query_embedding.reshape(1, -1).astype('float32')
    search_k = top_k + 1 if exclude_index is not None else top_k
    similarities, indices = faiss_index.search(query_vector, search_k)
    
    retrieved_indices = []
    retrieved_sims = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx == -1:
            continue
        if exclude_index is not None and idx == exclude_index:
            continue
        retrieved_indices.append(int(idx))
        retrieved_sims.append(float(sim))
        if len(retrieved_indices) == top_k:
            break
    return retrieved_indices[:top_k], retrieved_sims[:top_k]

def retrieve_l2(query_embedding, faiss_index, top_k=10, exclude_index=None):
    """
    Retrieves Top-K nearest neighbors from the FAISS IndexFlatL2 index.
    """
    query_vector = query_embedding.reshape(1, -1).astype('float32')
    search_k = top_k + 1 if exclude_index is not None else top_k
    distances, indices = faiss_index.search(query_vector, search_k)
    
    retrieved_indices = []
    for idx in indices[0]:
        if idx == -1:
            continue
        if exclude_index is not None and idx == exclude_index:
            continue
        retrieved_indices.append(int(idx))
        if len(retrieved_indices) == top_k:
            break
    return retrieved_indices[:top_k]

def compute_metrics_for_mode(query_embeddings, labels, faiss_index, is_cosine=True):
    """
    Computes Precision@5, Recall@5, F1@5, mAP@5, and mAP@10.
    """
    precisions_5 = []
    recalls_5 = []
    f1s_5 = []
    aps_5 = []
    aps_10 = []
    
    num_queries = len(query_embeddings)
    for i in range(num_queries):
        query_emb = query_embeddings[i]
        query_label = labels[i]
        
        # Search for top-10 nearest neighbors
        if is_cosine:
            retrieved_idxs, _ = retrieve_ip(query_emb, faiss_index, top_k=10, exclude_index=i)
        else:
            retrieved_idxs = retrieve_l2(query_emb, faiss_index, top_k=10, exclude_index=i)
            
        class_count = np.sum(labels == query_label)
        total_relevant = class_count - 1  # 625 - 1 = 624
        
        if total_relevant <= 0:
            continue
            
        # Relevancy indicators for the retrieved items
        rel = [int(labels[idx] == query_label) for idx in retrieved_idxs]
        
        # Compute precision at each rank (1-indexed)
        p_at_rank = []
        for rank in range(1, len(retrieved_idxs) + 1):
            p_at_rank.append(sum(rel[:rank]) / rank)
            
        # If retrieved size is less than 10 (pad with 0s)
        while len(p_at_rank) < 10:
            p_at_rank.append(0.0)
            rel.append(0)
            
        # AP@5: Average Precision at rank 5
        ap_5 = sum(p_at_rank[j] * rel[j] for j in range(5)) / 5.0
        
        # AP@10: Average Precision at rank 10
        ap_10 = sum(p_at_rank[j] * rel[j] for j in range(10)) / 10.0
        
        # Metrics@5
        rel_5 = rel[:5]
        num_rel_5 = sum(rel_5)
        prec_5 = num_rel_5 / 5.0
        rec_5 = num_rel_5 / total_relevant
        f1_5 = (2 * prec_5 * rec_5) / (prec_5 + rec_5) if (prec_5 + rec_5) > 0 else 0.0
        
        precisions_5.append(prec_5)
        recalls_5.append(rec_5)
        f1s_5.append(f1_5)
        aps_5.append(ap_5)
        aps_10.append(ap_10)
        
    return {
        "precision_5": float(np.mean(precisions_5)),
        "recall_5": float(np.mean(recalls_5)),
        "f1_5": float(np.mean(f1s_5)),
        "map_5": float(np.mean(aps_5)),
        "map_10": float(np.mean(aps_10))
    }

def main():
    labels_file = os.path.join(EMBEDDING_DIR, "labels.npy")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file '{labels_file}' not found! Run extraction scripts first.")
        
    labels = np.load(labels_file)
    print("Pre-computing evaluation metrics for all configurations...")

    # Load Baseline Embeddings and FAISS indices
    print("Loading Baseline Embeddings...")
    b_pan_embs = np.load(os.path.join(EMBEDDING_DIR, "pan_embeddings.npy"))
    b_mul_embs = np.load(os.path.join(EMBEDDING_DIR, "mul_embeddings.npy"))
    b_pan_index = faiss.read_index(os.path.join(FAISS_DIR, "pan_index.bin"))
    b_mul_index = faiss.read_index(os.path.join(FAISS_DIR, "mul_index.bin"))

    # Load Contrastive Embeddings and FAISS indices
    print("Loading Contrastive Embeddings...")
    c_pan_embs = np.load(os.path.join(EMBEDDING_DIR, "pan_embeddings_contrastive.npy"))
    c_mul_embs = np.load(os.path.join(EMBEDDING_DIR, "mul_embeddings_contrastive.npy"))
    c_pan_index = faiss.read_index(os.path.join(FAISS_DIR, "pan_index_contrastive.bin"))
    c_mul_index = faiss.read_index(os.path.join(FAISS_DIR, "mul_index_contrastive.bin"))

    summary = {
        "baseline": {},
        "contrastive": {}
    }

    # 1. Compute Baseline Metrics (512D L2)
    print("\nComputing Baseline Metrics...")
    summary["baseline"]["PAN_PAN"] = compute_metrics_for_mode(b_pan_embs, labels, b_pan_index, is_cosine=False)
    summary["baseline"]["MUL_MUL"] = compute_metrics_for_mode(b_mul_embs, labels, b_mul_index, is_cosine=False)
    summary["baseline"]["PAN_MUL"] = compute_metrics_for_mode(b_pan_embs, labels, b_mul_index, is_cosine=False)
    summary["baseline"]["MUL_PAN"] = compute_metrics_for_mode(b_mul_embs, labels, b_pan_index, is_cosine=False)

    # 2. Compute Contrastive Metrics (128D Cosine Similarity)
    print("\nComputing Contrastive Metrics...")
    summary["contrastive"]["PAN_PAN"] = compute_metrics_for_mode(c_pan_embs, labels, c_pan_index, is_cosine=True)
    summary["contrastive"]["MUL_MUL"] = compute_metrics_for_mode(c_mul_embs, labels, c_mul_index, is_cosine=True)
    summary["contrastive"]["PAN_MUL"] = compute_metrics_for_mode(c_pan_embs, labels, c_mul_index, is_cosine=True)
    summary["contrastive"]["MUL_PAN"] = compute_metrics_for_mode(c_mul_embs, labels, c_pan_index, is_cosine=True)

    # 3. Save to separate JSON files
    baseline_file = os.path.join(LOG_DIR, "metrics_summary_baseline.json")
    with open(baseline_file, "w") as f:
        json.dump(summary["baseline"], f, indent=4)
    save_versioned_file(baseline_file)

    contrastive_file = os.path.join(LOG_DIR, "metrics_summary_contrastive.json")
    with open(contrastive_file, "w") as f:
        json.dump(summary["contrastive"], f, indent=4)
    save_versioned_file(contrastive_file)
        
    print(f"\nSuccessfully pre-computed all metrics and saved to separate files under '{LOG_DIR}'!")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
