import os
import time
import json
import numpy as np
import faiss
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DSRSIDDataset
from train_contrastive import ContrastiveModel
from config import (
    DATASET_PATH,
    MODEL_DIR,
    EMBEDDING_DIR,
    FAISS_DIR,
    LOG_DIR,
    BATCH_SIZE,
    FREEZE_BACKBONE,
    FAISS_INDEX,
    IS_COLAB,
    save_versioned_file,
    create_faiss_index,
    check_safeguard,
    verify_cache,
    update_pipeline_manifest
)

def evaluate_vectorized(query_embeddings, labels, faiss_index, top_k=5, exclude_self=True):
    """
    Vectorized evaluation of Precision@K, Recall@K, and F1@K.
    Runs batch queries in FAISS and computes metrics using NumPy vectorized operations.
    """
    # Ensure float32
    query_embeddings = query_embeddings.astype('float32')
    num_queries = len(query_embeddings)
    
    # Search for top_k + 1 to exclude self-match
    search_k = top_k + 1 if exclude_self else top_k
    
    # Measure FAISS search time
    start_time = time.perf_counter()
    distances, indices = faiss_index.search(query_embeddings, search_k)
    search_time = time.perf_counter() - start_time
    
    # Exclude the query's corresponding index
    filtered_indices = np.zeros((num_queries, top_k), dtype=np.int64)
    if exclude_self:
        query_indices = np.arange(num_queries).reshape(-1, 1)
        is_self = (indices == query_indices)
        for i in range(num_queries):
            row_idx = indices[i]
            row_self = is_self[i]
            non_self = row_idx[~row_self]
            filtered_indices[i] = non_self[:top_k]
    else:
        filtered_indices = indices[:, :top_k]
        
    # Get labels
    retrieved_labels = labels[filtered_indices]
    query_labels = labels.reshape(-1, 1)
    
    # Computes matches
    rel = (retrieved_labels == query_labels).astype(np.float32)
    
    # Precision@K
    precisions = np.sum(rel, axis=1) / top_k
    
    # Recall@K
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    label_to_count = dict(zip(unique_labels, class_counts))
    
    total_relevant = np.array([label_to_count[l] for l in labels], dtype=np.float32)
    if exclude_self:
        total_relevant = total_relevant - 1.0
        
    total_relevant = np.maximum(total_relevant, 1.0)
    recalls = np.sum(rel, axis=1) / total_relevant
    
    # F1-Score
    sum_pr = precisions + recalls
    f1s = np.zeros_like(precisions)
    non_zero_mask = sum_pr > 0
    f1s[non_zero_mask] = (2 * precisions[non_zero_mask] * recalls[non_zero_mask]) / sum_pr[non_zero_mask]
    
    return np.mean(precisions), np.mean(recalls), np.mean(f1s), search_time

def main():
    check_safeguard()
    
    best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
    indices_path = os.path.join(EMBEDDING_DIR, "subset_indices.npy")
    
    if not os.path.exists(best_model_path) or not os.path.exists(indices_path):
        raise FileNotFoundError(
            "Trained model or indices files not found! Please run 'train_contrastive.py' first."
        )

    pan_emb_path = os.path.join(EMBEDDING_DIR, "pan_embeddings_contrastive.npy")
    mul_emb_path = os.path.join(EMBEDDING_DIR, "mul_embeddings_contrastive.npy")
    pan_idx_path = os.path.join(FAISS_DIR, "pan_index_contrastive.bin")
    mul_idx_path = os.path.join(FAISS_DIR, "mul_index_contrastive.bin")

    # --- STAGE A: Contrastive Embeddings ---
    print("Checking Contrastive Embeddings...")
    is_cached_emb, reason_emb = verify_cache("contrastive_embeddings")
    
    if is_cached_emb:
        print("✔ Contrastive embeddings cache found. Skipping extraction.\n")
        pan_embs_norm = np.load(pan_emb_path)
        mul_embs_norm = np.load(mul_emb_path)
    else:
        if reason_emb:
            print(f"⚠ Cache invalid. Reason: {reason_emb}")
        
        # Load indices and setup dataset/dataloader
        subset_indices = np.load(indices_path)
        dataset = DSRSIDDataset(file_path=DATASET_PATH, indices=subset_indices)
        
        # Dynamic workers based on OS
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_workers = 2 if (IS_COLAB and device.type == 'cuda') else 0
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(num_workers > 0)
        )
        
        # Instantiate and load contrastive model
        print("Loading contrastive model...")
        model = ContrastiveModel(freeze_backbone=FREEZE_BACKBONE)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        model = model.to(device)

        # Extract 128-dimensional contrastive embeddings
        pan_embs_list = []
        mul_embs_list = []

        print("Extracting 128D contrastive embeddings...")
        use_amp = (device.type == 'cuda')
        with torch.no_grad():
            for pan_batch, mul_batch, _ in tqdm(dataloader, desc="Inference Batches"):
                pan_batch = pan_batch.to(device, non_blocking=True)
                mul_batch = mul_batch.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    pan_proj = model.forward_pan(pan_batch)
                    mul_proj = model.forward_mul(mul_batch)
                
                pan_embs_list.append(pan_proj.cpu().numpy())
                mul_embs_list.append(mul_proj.cpu().numpy())

        pan_embs = np.concatenate(pan_embs_list, axis=0)
        mul_embs = np.concatenate(mul_embs_list, axis=0)

        # L2 Normalize embeddings (critical for cosine similarity match in FAISS)
        print("Normalizing embeddings for Cosine Similarity...")
        pan_embs_norm = pan_embs / np.linalg.norm(pan_embs, axis=1, keepdims=True)
        mul_embs_norm = mul_embs / np.linalg.norm(mul_embs, axis=1, keepdims=True)

        # Save embeddings
        np.save(pan_emb_path, pan_embs_norm)
        np.save(mul_emb_path, mul_embs_norm)
        save_versioned_file(pan_emb_path)
        save_versioned_file(mul_emb_path)
        print("Contrastive embeddings saved to disk.")
        update_pipeline_manifest("contrastive_embeddings", True)
        dataset.close()

    # Load labels
    labels = np.load(os.path.join(EMBEDDING_DIR, "labels.npy"))

    # --- STAGE B: Contrastive FAISS Indices ---
    print("Checking Contrastive FAISS Indices...")
    is_faiss_cached, reason_faiss = verify_cache("faiss_ready")
    contrastive_indices_exist = os.path.exists(pan_idx_path) and os.path.exists(mul_idx_path)
    
    if is_faiss_cached or (contrastive_indices_exist and verify_cache("contrastive_embeddings")[0]):
        print("✔ Contrastive FAISS indices are already built and valid. Skipping building.\n")
        pan_index_contrastive = faiss.read_index(pan_idx_path)
        mul_index_contrastive = faiss.read_index(mul_idx_path)
        
        def get_index_type_str(index):
            class_name = index.__class__.__name__
            if "FlatL2" in class_name: return "FlatL2"
            if "FlatIP" in class_name: return "FlatIP"
            if "HNSW" in class_name: return "HNSW"
            if "IVFFlat" in class_name: return "IVF Flat"
            if "IVFPQ" in class_name: return "IVF PQ"
            return FAISS_INDEX
            
        pan_resolved = get_index_type_str(pan_index_contrastive)
        mul_resolved = get_index_type_str(mul_index_contrastive)
        pan_build_time = 0.0
        mul_build_time = 0.0
    else:
        if reason_faiss and not contrastive_indices_exist:
            print(f"⚠ Cache invalid. Reason: {reason_faiss}")
        print("Building contrastive FAISS indices...")
        
        # Build FAISS indices using dynamic index builder (Cosine uses INNER_PRODUCT metric)
        print("\nBuilding contrastive FAISS index for PAN modality...")
        pan_index_contrastive, pan_resolved, pan_build_time = create_faiss_index(
            pan_embs_norm, 
            index_type=FAISS_INDEX, 
            metric_type=faiss.METRIC_INNER_PRODUCT
        )

        print("\nBuilding contrastive FAISS index for MUL modality...")
        mul_index_contrastive, mul_resolved, mul_build_time = create_faiss_index(
            mul_embs_norm, 
            index_type=FAISS_INDEX, 
            metric_type=faiss.METRIC_INNER_PRODUCT
        )

        faiss.write_index(pan_index_contrastive, pan_idx_path)
        faiss.write_index(mul_index_contrastive, mul_idx_path)
        save_versioned_file(pan_idx_path)
        save_versioned_file(mul_idx_path)
        print("Contrastive FAISS indices saved.")

    # 6. Evaluate Contrastive Model (vectorized and benchmarked)
    print("\nEvaluating Contrastive Model retrieval performance...")
    c_pan_pan_p, c_pan_pan_r, c_pan_pan_f, t_pan_pan = evaluate_vectorized(pan_embs_norm, labels, pan_index_contrastive, exclude_self=True)
    c_mul_mul_p, c_mul_mul_r, c_mul_mul_f, t_mul_mul = evaluate_vectorized(mul_embs_norm, labels, mul_index_contrastive, exclude_self=True)
    c_pan_mul_p, c_pan_mul_r, c_pan_mul_f, t_pan_mul = evaluate_vectorized(pan_embs_norm, labels, mul_index_contrastive, exclude_self=True)
    c_mul_pan_p, c_mul_pan_r, c_mul_pan_f, t_mul_pan = evaluate_vectorized(mul_embs_norm, labels, pan_index_contrastive, exclude_self=True)

    # 7. Evaluate Baseline Model (vectorized and benchmarked)
    print("\nEvaluating Baseline Model retrieval performance for comparison...")
    base_pan_path = os.path.join(EMBEDDING_DIR, "pan_embeddings.npy")
    base_mul_path = os.path.join(EMBEDDING_DIR, "mul_embeddings.npy")
    base_pan = np.load(base_pan_path, mmap_mode='r')
    base_mul = np.load(base_mul_path, mmap_mode='r')
    
    # Auto-run build_faiss.py if baseline indices are missing
    base_pan_idx_file = os.path.join(FAISS_DIR, "pan_index.bin")
    base_mul_idx_file = os.path.join(FAISS_DIR, "mul_index.bin")
    if not os.path.exists(base_pan_idx_file) or not os.path.exists(base_mul_idx_file):
        print("Baseline FAISS indices not found. Automatically running build_faiss.py to generate them...")
        import subprocess
        import sys
        cmd = [sys.executable, "-X", "utf8", "build_faiss.py"]
        subprocess.run(cmd, check=True)
        
    base_pan_index = faiss.read_index(base_pan_idx_file)
    base_mul_index = faiss.read_index(base_mul_idx_file)

    # Baseline uses L2 metric, which is exact/approx depending on baseline index build
    b_pan_pan_p, b_pan_pan_r, b_pan_pan_f, tb_pan_pan = evaluate_vectorized(base_pan, labels, base_pan_index, exclude_self=True)
    b_mul_mul_p, b_mul_mul_r, b_mul_mul_f, tb_mul_mul = evaluate_vectorized(base_mul, labels, base_mul_index, exclude_self=True)
    b_pan_mul_p, b_pan_mul_r, b_pan_mul_f, tb_pan_mul = evaluate_vectorized(base_pan, labels, base_mul_index, exclude_self=True)
    b_mul_pan_p, b_mul_pan_r, b_mul_pan_f, tb_mul_pan = evaluate_vectorized(base_mul, labels, base_pan_index, exclude_self=True)

    # 8. Print tabular comparison report
    print("\n" + "="*80)
    print("                      RETRIEVAL PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Retrieval Mode':<18} | {'Metric':<10} | {'Baseline (L2)':<20} | {'Contrastive (Cosine)':<22} | {'Improvement':<12}")
    print("-"*80)

    modes = [
        ("PAN -> PAN", (b_pan_pan_p, c_pan_pan_p), (b_pan_pan_r, c_pan_pan_r), (b_pan_pan_f, c_pan_pan_f)),
        ("MUL -> MUL", (b_mul_mul_p, c_mul_mul_p), (b_mul_mul_r, c_mul_mul_r), (b_mul_mul_f, c_mul_mul_f)),
        ("PAN -> MUL", (b_pan_mul_p, c_pan_mul_p), (b_pan_mul_r, c_pan_mul_r), (b_pan_mul_f, c_pan_mul_f)),
        ("MUL -> PAN", (b_mul_pan_p, c_mul_pan_p), (b_mul_pan_r, c_mul_pan_r), (b_mul_pan_f, c_mul_pan_f)),
    ]

    for mode_name, p_vals, r_vals, f_vals in modes:
        p_diff = p_vals[1] - p_vals[0]
        print(f"{mode_name:<18} | {'Precision':<10} | {p_vals[0]:<20.5f} | {p_vals[1]:<22.5f} | {p_diff:+.5f}")
        r_diff = r_vals[1] - r_vals[0]
        print(f"{'':<18} | {'Recall':<10} | {r_vals[0]:<20.5f} | {r_vals[1]:<22.5f} | {r_diff:+.5f}")
        f_diff = f_vals[1] - f_vals[0]
        print(f"{'':<18} | {'F1-Score':<10} | {f_vals[0]:<20.5f} | {f_vals[1]:<22.5f} | {f_diff:+.5f}")
        print("-"*80)
    print("="*80)

    # 9. Retrieval Latency & QPS Benchmarking
    num_queries = len(labels)
    avg_latency = (t_pan_mul / num_queries) * 1000.0  # ms per query
    qps = num_queries / t_pan_mul

    print(f"\n--- Engineering Retrieval Benchmark ({mul_resolved}) ---")
    print(f"Total Queries Evaluated: {num_queries}")
    print(f"Average Search Latency:   {avg_latency:.4f} ms/query")
    print(f"Queries Per Second (QPS): {qps:.2f} queries/sec")
    print(f"Index Build Time (PAN):   {pan_build_time:.3f} s")
    print(f"Index Build Time (MUL):   {mul_build_time:.3f} s")
    
    pan_index_size = os.path.getsize(pan_idx_path)
    mul_index_size = os.path.getsize(mul_idx_path)
    print(f"Index File Size (PAN):    {pan_index_size / (1024**2):.3f} MB")
    print(f"Index File Size (MUL):    {mul_index_size / (1024**2):.3f} MB")

    # Save benchmark stats to json
    benchmark_data = {
        "num_queries": num_queries,
        "avg_search_latency_ms": avg_latency,
        "queries_per_second": qps,
        "pan_build_time_sec": pan_build_time,
        "mul_build_time_sec": mul_build_time,
        "pan_index_file_size_bytes": pan_index_size,
        "mul_index_file_size_bytes": mul_index_size,
        "pan_resolved_index_type": pan_resolved,
        "mul_resolved_index_type": mul_resolved
    }
    
    benchmark_file = os.path.join(LOG_DIR, "retrieval_benchmark.json")
    with open(benchmark_file, "w") as bf:
        json.dump(benchmark_data, bf, indent=4)
    save_versioned_file(benchmark_file)

    # Save experiment metadata
    from config import save_experiment_metadata
    save_experiment_metadata(resolved_faiss_index=mul_resolved)

    if 'dataset' in locals():
        dataset.close()
    
    # Save stage state
    # We update faiss_ready to True in manifest ONLY if baseline index files exist as well
    base_pan_idx = os.path.join(FAISS_DIR, "pan_index.bin")
    base_mul_idx = os.path.join(FAISS_DIR, "mul_index.bin")
    if os.path.exists(base_pan_idx) and os.path.exists(base_mul_idx) and os.path.exists(pan_idx_path) and os.path.exists(mul_idx_path):
        update_pipeline_manifest("faiss_ready", True)

if __name__ == "__main__":
    main()
