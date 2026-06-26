import os
import numpy as np
import faiss
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DSRSIDDataset
from train_contrastive import ContrastiveModel
from config import DATASET_PATH, MODEL_DIR, EMBEDDING_DIR, FAISS_DIR, BATCH_SIZE, FREEZE_BACKBONE, save_versioned_file

def retrieve_ip(query_embedding, faiss_index, top_k=5, exclude_index=None):
    """
    Retrieves Top-K nearest neighbors from the FAISS IndexFlatIP (Cosine Similarity) index.
    """
    query_vector = query_embedding.reshape(1, -1).astype('float32')
    search_k = top_k + 1 if exclude_index is not None else top_k
    
    # search returns (similarities, indices)
    similarities, indices = faiss_index.search(query_vector, search_k)
    
    retrieved_indices = []
    retrieved_dists = []
    
    for sim, idx in zip(similarities[0], indices[0]):
        if idx == -1:
            continue
        if exclude_index is not None and idx == exclude_index:
            continue
        retrieved_indices.append(int(idx))
        retrieved_dists.append(float(sim))
        if len(retrieved_indices) == top_k:
            break
            
    return retrieved_indices[:top_k], retrieved_dists[:top_k]

def retrieve_l2(query_embedding, faiss_index, top_k=5, exclude_index=None):
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

def evaluate(query_embeddings, labels, faiss_index, is_cosine=True, top_k=5):
    """
    Evaluates Precision@5, Recall@5, and F1@5 across the dataset.
    """
    precisions = []
    recalls = []
    f1s = []
    
    num_queries = len(query_embeddings)
    for i in range(num_queries):
        query_emb = query_embeddings[i]
        query_label = labels[i]
        
        # Perform retrieval
        if is_cosine:
            retrieved_idxs, _ = retrieve_ip(query_emb, faiss_index, top_k=top_k, exclude_index=i)
        else:
            retrieved_idxs = retrieve_l2(query_emb, faiss_index, top_k=top_k, exclude_index=i)
            
        class_count = np.sum(labels == query_label)
        total_relevant = class_count - 1 # exclude the query's matching pair
        
        if total_relevant <= 0:
            continue
            
        num_relevant_retrieved = sum(labels[idx] == query_label for idx in retrieved_idxs)
        
        p = num_relevant_retrieved / top_k
        r = num_relevant_retrieved / total_relevant
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

def main():
    best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
    indices_path = os.path.join(EMBEDDING_DIR, "subset_indices.npy")
    
    if not os.path.exists(best_model_path) or not os.path.exists(indices_path):
        raise FileNotFoundError(
            "Trained model or indices files not found! Please run 'train_contrastive.py' first."
        )

    # 1. Load indices and setup dataset/dataloader
    subset_indices = np.load(indices_path)
    dataset = DSRSIDDataset(file_path=DATASET_PATH, indices=subset_indices)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    labels = np.load(os.path.join(EMBEDDING_DIR, "labels.npy"))

    # 2. Instantiate and load contrastive model
    print("Loading contrastive model...")
    model = ContrastiveModel(freeze_backbone=FREEZE_BACKBONE)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 3. Extract 128-dimensional contrastive embeddings
    pan_embs_list = []
    mul_embs_list = []

    print("Extracting 128D contrastive embeddings...")
    with torch.no_grad():
        for pan_batch, mul_batch, _ in tqdm(dataloader, desc="Batches"):
            pan_batch = pan_batch.to(device)
            mul_batch = mul_batch.to(device)
            
            pan_proj = model.forward_pan(pan_batch)
            mul_proj = model.forward_mul(mul_batch)
            
            pan_embs_list.append(pan_proj.cpu().numpy())
            mul_embs_list.append(mul_proj.cpu().numpy())

    pan_embs = np.concatenate(pan_embs_list, axis=0)
    mul_embs = np.concatenate(mul_embs_list, axis=0)

    # 4. L2 Normalize embeddings (critical for cosine similarity match in FAISS IndexFlatIP)
    print("Normalizing embeddings for Cosine Similarity...")
    pan_embs_norm = pan_embs / np.linalg.norm(pan_embs, axis=1, keepdims=True)
    mul_embs_norm = mul_embs / np.linalg.norm(mul_embs, axis=1, keepdims=True)

    # Save embeddings
    pan_emb_path = os.path.join(EMBEDDING_DIR, "pan_embeddings_contrastive.npy")
    mul_emb_path = os.path.join(EMBEDDING_DIR, "mul_embeddings_contrastive.npy")
    np.save(pan_emb_path, pan_embs_norm)
    np.save(mul_emb_path, mul_embs_norm)
    save_versioned_file(pan_emb_path)
    save_versioned_file(mul_emb_path)
    print("Contrastive embeddings saved to disk.")

    # 5. Build FAISS IndexFlatIP indices (Cosine similarity)
    dimension = 128
    pan_index_contrastive = faiss.IndexFlatIP(dimension)
    mul_index_contrastive = faiss.IndexFlatIP(dimension)

    pan_index_contrastive.add(pan_embs_norm.astype('float32'))
    mul_index_contrastive.add(mul_embs_norm.astype('float32'))

    pan_idx_path = os.path.join(FAISS_DIR, "pan_index_contrastive.bin")
    mul_idx_path = os.path.join(FAISS_DIR, "mul_index_contrastive.bin")
    faiss.write_index(pan_index_contrastive, pan_idx_path)
    faiss.write_index(mul_index_contrastive, mul_idx_path)
    save_versioned_file(pan_idx_path)
    save_versioned_file(mul_idx_path)
    print("Contrastive FAISS indices built and saved.")

    # 6. Evaluate Contrastive Model (128D Cosine)
    print("\nEvaluating Contrastive Model retrieval performance...")
    c_pan_pan_p, c_pan_pan_r, c_pan_pan_f = evaluate(pan_embs_norm, labels, pan_index_contrastive, is_cosine=True)
    c_mul_mul_p, c_mul_mul_r, c_mul_mul_f = evaluate(mul_embs_norm, labels, mul_index_contrastive, is_cosine=True)
    c_pan_mul_p, c_pan_mul_r, c_pan_mul_f = evaluate(pan_embs_norm, labels, mul_index_contrastive, is_cosine=True)
    c_mul_pan_p, c_mul_pan_r, c_mul_pan_f = evaluate(mul_embs_norm, labels, pan_index_contrastive, is_cosine=True)

    # 7. Evaluate Baseline Model (512D L2) dynamically for side-by-side comparison
    print("Evaluating Baseline Model retrieval performance for comparison...")
    base_pan = np.load(os.path.join(EMBEDDING_DIR, "pan_embeddings.npy"))
    base_mul = np.load(os.path.join(EMBEDDING_DIR, "mul_embeddings.npy"))
    base_pan_index = faiss.read_index(os.path.join(FAISS_DIR, "pan_index.bin"))
    base_mul_index = faiss.read_index(os.path.join(FAISS_DIR, "mul_index.bin"))

    b_pan_pan_p, b_pan_pan_r, b_pan_pan_f = evaluate(base_pan, labels, base_pan_index, is_cosine=False)
    b_mul_mul_p, b_mul_mul_r, b_mul_mul_f = evaluate(base_mul, labels, base_mul_index, is_cosine=False)
    b_pan_mul_p, b_pan_mul_r, b_pan_mul_f = evaluate(base_pan, labels, base_mul_index, is_cosine=False)
    b_mul_pan_p, b_mul_pan_r, b_mul_pan_f = evaluate(base_mul, labels, base_pan_index, is_cosine=False)

    # 8. Print tabular comparison report
    print("\n" + "="*80)
    print("                      RETRIEVAL PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Retrieval Mode':<18} | {'Metric':<10} | {'Baseline (512D L2)':<20} | {'Contrastive (128D Cos)':<22} | {'Improvement':<12}")
    print("-"*80)

    modes = [
        ("PAN -> PAN", (b_pan_pan_p, c_pan_pan_p), (b_pan_pan_r, c_pan_pan_r), (b_pan_pan_f, c_pan_pan_f)),
        ("MUL -> MUL", (b_mul_mul_p, c_mul_mul_p), (b_mul_mul_r, c_mul_mul_r), (b_mul_mul_f, c_mul_mul_f)),
        ("PAN -> MUL", (b_pan_mul_p, c_pan_mul_p), (b_pan_mul_r, c_pan_mul_r), (b_pan_mul_f, c_pan_mul_f)),
        ("MUL -> PAN", (b_mul_pan_p, c_mul_pan_p), (b_mul_pan_r, c_mul_pan_r), (b_mul_pan_f, c_mul_pan_f)),
    ]

    for mode_name, p_vals, r_vals, f_vals in modes:
        # Precision row
        p_diff = p_vals[1] - p_vals[0]
        print(f"{mode_name:<18} | {'Precision':<10} | {p_vals[0]:<20.5f} | {p_vals[1]:<22.5f} | {p_diff:+.5f}")
        # Recall row
        r_diff = r_vals[1] - r_vals[0]
        print(f"{'':<18} | {'Recall':<10} | {r_vals[0]:<20.5f} | {r_vals[1]:<22.5f} | {r_diff:+.5f}")
        # F1 row
        f_diff = f_vals[1] - f_vals[0]
        print(f"{'':<18} | {'F1-Score':<10} | {f_vals[0]:<20.5f} | {f_vals[1]:<22.5f} | {f_diff:+.5f}")
        print("-"*80)

    print("="*80)

    dataset.close()

if __name__ == "__main__":
    main()
