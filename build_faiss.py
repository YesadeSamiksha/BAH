import os
import time
import numpy as np
import faiss
from config import (
    EMBEDDING_DIR,
    FAISS_DIR,
    FAISS_INDEX,
    save_versioned_file,
    create_faiss_index,
    verify_cache,
    update_pipeline_manifest
)

def main():
    pan_embeddings_path = os.path.join(EMBEDDING_DIR, "pan_embeddings.npy")
    mul_embeddings_path = os.path.join(EMBEDDING_DIR, "mul_embeddings.npy")

    # 1. Skip Check
    pan_index_file = os.path.join(FAISS_DIR, "pan_index.bin")
    mul_index_file = os.path.join(FAISS_DIR, "mul_index.bin")
    
    print("Checking Baseline FAISS Indices...")
    is_faiss_ready, _ = verify_cache("faiss_ready")
    is_baseline_valid, _ = verify_cache("baseline_embeddings")
    baseline_indices_exist = os.path.exists(pan_index_file) and os.path.exists(mul_index_file)
    
    if is_faiss_ready or (is_baseline_valid and baseline_indices_exist):
        print("✔ Baseline FAISS indices are already built and valid. Skipping.\n")
        return

    # 2. Load the extracted baseline embeddings using memmap for memory efficiency
    if not os.path.exists(pan_embeddings_path) or not os.path.exists(mul_embeddings_path):
        raise FileNotFoundError(
            "Embeddings files not found! Please run 'extract_embeddings.py' first."
        )

    print(f"Loading PAN embeddings (memmap) from: {pan_embeddings_path}...")
    pan_embeddings = np.load(pan_embeddings_path, mmap_mode='r')
    print(f"Loading MUL embeddings (memmap) from: {mul_embeddings_path}...")
    mul_embeddings = np.load(mul_embeddings_path, mmap_mode='r')

    print(f"PAN embeddings shape: {pan_embeddings.shape}")
    print(f"MUL embeddings shape: {mul_embeddings.shape}")

    # 3. Build baseline indices (using L2 metric)
    print("\nBuilding FAISS index for PAN modality...")
    pan_index, pan_resolved_type, pan_build_time = create_faiss_index(
        pan_embeddings, 
        index_type=FAISS_INDEX, 
        metric_type=faiss.METRIC_L2
    )

    print("\nBuilding FAISS index for MUL modality...")
    mul_index, mul_resolved_type, mul_build_time = create_faiss_index(
        mul_embeddings, 
        index_type=FAISS_INDEX, 
        metric_type=faiss.METRIC_L2
    )

    # 4. Save the built indices to disk
    print(f"\nSaving PAN FAISS index to: {pan_index_file}...")
    faiss.write_index(pan_index, pan_index_file)
    save_versioned_file(pan_index_file)
    
    print(f"Saving MUL FAISS index to: {mul_index_file}...")
    faiss.write_index(mul_index, mul_index_file)
    save_versioned_file(mul_index_file)

    # 5. Measure size on disk
    pan_size = os.path.getsize(pan_index_file)
    mul_size = os.path.getsize(mul_index_file)

    print("\nFAISS Indexing completed successfully!")
    print(f"PAN Index size: {pan_size / (1024**2):.3f} MB | Build Time: {pan_build_time:.3f} s | Resolved Type: {pan_resolved_type}")
    print(f"MUL Index size: {mul_size / (1024**2):.3f} MB | Build Time: {mul_build_time:.3f} s | Resolved Type: {mul_resolved_type}")

    # Save experiment metadata
    from config import save_experiment_metadata
    save_experiment_metadata(resolved_faiss_index=mul_resolved_type)

    # 6. Save stage state
    # We set faiss_ready to True in manifest ONLY if contrastive index files also exist
    pan_idx_c = os.path.join(FAISS_DIR, "pan_index_contrastive.bin")
    mul_idx_c = os.path.join(FAISS_DIR, "mul_index_contrastive.bin")
    if os.path.exists(pan_idx_c) and os.path.exists(mul_idx_c):
        update_pipeline_manifest("faiss_ready", True)
    else:
        from config import get_pipeline_manifest, save_pipeline_manifest
        manifest = get_pipeline_manifest()
        manifest["stages"]["faiss_ready"] = False
        save_pipeline_manifest(manifest)

if __name__ == "__main__":
    main()
