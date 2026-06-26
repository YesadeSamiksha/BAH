import os
import numpy as np
import faiss

def main():
    pan_embeddings_path = "pan_embeddings.npy"
    mul_embeddings_path = "mul_embeddings.npy"

    # 1. Load the extracted embeddings
    if not os.path.exists(pan_embeddings_path) or not os.path.exists(mul_embeddings_path):
        raise FileNotFoundError(
            "Embeddings files not found! Please run 'extract_embeddings.py' first."
        )

    print(f"Loading PAN embeddings from: {pan_embeddings_path}...")
    pan_embeddings = np.load(pan_embeddings_path)
    print(f"Loading MUL embeddings from: {mul_embeddings_path}...")
    mul_embeddings = np.load(mul_embeddings_path)

    # FAISS requires float32 datatype
    pan_embeddings = pan_embeddings.astype('float32')
    mul_embeddings = mul_embeddings.astype('float32')

    print(f"PAN embeddings shape: {pan_embeddings.shape}")
    print(f"MUL embeddings shape: {mul_embeddings.shape}")

    dimension = pan_embeddings.shape[1]  # 512

    # 2. Build FAISS IndexFlatL2 for PAN modality
    print("\nBuilding FAISS index for PAN modality...")
    pan_index = faiss.IndexFlatL2(dimension)
    pan_index.add(pan_embeddings)
    print(f"PAN index trained: {pan_index.is_trained}")
    print(f"Total vectors in PAN index: {pan_index.ntotal}")

    # 3. Build FAISS IndexFlatL2 for MUL modality
    print("Building FAISS index for MUL modality...")
    mul_index = faiss.IndexFlatL2(dimension)
    mul_index.add(mul_embeddings)
    print(f"MUL index trained: {mul_index.is_trained}")
    print(f"Total vectors in MUL index: {mul_index.ntotal}")

    # 4. Save the built indices to disk
    pan_index_file = "pan_index.bin"
    mul_index_file = "mul_index.bin"
    
    print(f"\nSaving PAN FAISS index to: {pan_index_file}...")
    faiss.write_index(pan_index, pan_index_file)
    print(f"Saving MUL FAISS index to: {mul_index_file}...")
    faiss.write_index(mul_index, mul_index_file)

    print("\nFAISS Indexing completed successfully!")

if __name__ == "__main__":
    main()
