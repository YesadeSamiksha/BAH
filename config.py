import os
import sys
import shutil
import datetime

# Detect if running on Google Colab
IS_COLAB = False
try:
    import google.colab
    IS_COLAB = True
except ImportError:
    pass

# Mount Google Drive automatically if on Colab
if IS_COLAB:
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount("/content/drive")
    except Exception as e:
        print(f"Warning: Failed to mount Google Drive: {e}")

# Setup Roots
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if IS_COLAB:
    STORAGE_ROOT = "/content/drive/MyDrive/BAH"
else:
    STORAGE_ROOT = PROJECT_ROOT

# Automatic Dataset Discovery
possible_paths = [
    os.path.join(STORAGE_ROOT, "data", "DSRSID.mat"),
    os.path.join(STORAGE_ROOT, "dataset", "DSRSID.mat")
]

DATASET_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        DATASET_PATH = path
        break

if DATASET_PATH is None:
    checked_paths_str = "\n".join(possible_paths)
    raise FileNotFoundError(
        f"Dataset not found.\n\nChecked:\n\n{checked_paths_str}\n\n"
        f"Please place DSRSID.mat into one of these directories."
    )

print(f"Using dataset:\n\n{DATASET_PATH}")

BACKBONE = "resnet50"  # Supported: "resnet18", "resnet50"
FAISS_INDEX = "auto"   # Supported: "auto", "FlatL2", "FlatIP", "IVF Flat", "IVF PQ", "HNSW"
VALIDATION_INTERVAL = 5
FREEZE_BACKBONE = True

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
TEMPERATURE = 0.07
PATIENCE = 5
AUTO_RESUME = True
USE_FULL_DATASET = True 

FORCE_LOCAL_FULL_TRAIN = False
READ_ONLY_MODE = False

def get_backbone_feature_dim(backbone_name):
    """
    Dynamically returns the feature dimension of the selected backbone.
    Uses a fast-path for standard backbones to avoid importing PyTorch/torchvision
    in pure-NumPy/Sklearn scripts, preventing Windows DLL load conflicts (WinError 1114).
    """
    backbone_name = backbone_name.lower()
    if backbone_name == "resnet18":
        return 512
    elif backbone_name == "resnet50":
        return 2048
        
    # Dynamic lookup fallback for other/future backbones
    try:
        import torchvision.models as models
        if hasattr(models, backbone_name):
            model = getattr(models, backbone_name)()
            if hasattr(model, "fc"):
                return model.fc.in_features
            elif hasattr(model, "classifier"):
                if hasattr(model.classifier, "in_features"):
                    return model.classifier.in_features
                return model.classifier[-1].in_features
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    except Exception as e:
        raise ValueError(f"Error resolving backbone '{backbone_name}': {e}")

def check_safeguard():
    """
    Aborts training/extraction/evaluation if attempting to run the full dataset locally.
    """
    if USE_FULL_DATASET and not IS_COLAB and not FORCE_LOCAL_FULL_TRAIN:
        print("\n" + "!" * 80)
        print("🛑 EXECUTION ABORTED: SAFEGUARD TRIGGERED")
        print("!" * 80)
        print("You are attempting to process the FULL 80,000-image dataset on a local machine.")
        print("This computationally intensive run is intended for a Google Colab GPU environment.")
        print("To bypass this safeguard and run locally anyway, set:")
        print("  FORCE_LOCAL_FULL_TRAIN = True")
        print("in config.py.")
        print("!" * 80 + "\n")
        raise RuntimeError("Local execution with USE_FULL_DATASET=True blocked by safeguard.")

# Versioning Configuration
VERSION = "v1"

def get_experiment_dir():
    """
    Returns the versioned experiment directory based on backbone and index type.
    """
    return os.path.join(STORAGE_ROOT, "experiments", f"{BACKBONE}_{FAISS_INDEX.lower()}")

# Centralized Directories (routed inside get_experiment_dir())
def get_dirs():
    exp_dir = get_experiment_dir()
    dirs = {
        "MODEL_DIR": os.path.join(exp_dir, "models"),
        "CHECKPOINT_DIR": os.path.join(exp_dir, "checkpoints"),
        "EMBEDDING_DIR": os.path.join(exp_dir, "embeddings"),
        "FAISS_DIR": os.path.join(exp_dir, "faiss_indices"),
        "LOG_DIR": os.path.join(exp_dir, "logs"),
        "OUTPUT_DIR": os.path.join(exp_dir, "outputs")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

# Initialize and expose directory constants
dirs = get_dirs()
MODEL_DIR = dirs["MODEL_DIR"]
CHECKPOINT_DIR = dirs["CHECKPOINT_DIR"]
EMBEDDING_DIR = dirs["EMBEDDING_DIR"]
FAISS_DIR = dirs["FAISS_DIR"]
LOG_DIR = dirs["LOG_DIR"]
OUTPUT_DIR = dirs["OUTPUT_DIR"]

def save_versioned_file(src_path, version=None):
    """
    Saves a versioned copy of the file at src_path.
    E.g. pan_embeddings.npy -> pan_embeddings_v1.npy
    """
    if version is None:
        version = VERSION
    if not version or not os.path.exists(src_path):
        return
    base, ext = os.path.splitext(src_path)
    versioned_path = f"{base}_{version}{ext}"
    shutil.copy(src_path, versioned_path)
    print(f"Versioned copy saved: {versioned_path}")

def save_timestamped_file(src_path):
    """
    Saves a timestamped copy of the file at src_path.
    E.g. best_model.pth -> best_model_2026-06-26_1530.pth
    """
    if not os.path.exists(src_path):
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    base, ext = os.path.splitext(src_path)
    timestamped_path = f"{base}_{timestamp}{ext}"
    shutil.copy(src_path, timestamped_path)
    print(f"Timestamped copy saved: {timestamped_path}")

# Retrieval Mode
# Options:
# "baseline"
# "contrastive"
RETRIEVAL_MODE = "contrastive"

# Stage Skipping Helpers
def get_dataset_hash():
    if not DATASET_PATH or not os.path.exists(DATASET_PATH):
        return ""
    stat = os.stat(DATASET_PATH)
    return f"{int(stat.st_mtime)}_{stat.st_size}"

# Define Stage Output Artifacts
STAGE_OUTPUTS = {
    "dataset_ready": [
        DATASET_PATH
    ],
    "baseline_embeddings": [
        os.path.join(EMBEDDING_DIR, "pan_embeddings.npy"),
        os.path.join(EMBEDDING_DIR, "mul_embeddings.npy"),
        os.path.join(EMBEDDING_DIR, "labels.npy"),
        os.path.join(EMBEDDING_DIR, "subset_indices.npy")
    ],
    "training_complete": [
        os.path.join(MODEL_DIR, "best_model.pth"),
        os.path.join(MODEL_DIR, "best_model_metadata.json"),
        os.path.join(get_experiment_dir(), "experiment.json"),
        os.path.join(OUTPUT_DIR, "training_curve.png"),
        os.path.join(LOG_DIR, "train_loss.csv")
    ],
    "contrastive_embeddings": [
        os.path.join(EMBEDDING_DIR, "pan_embeddings_contrastive.npy"),
        os.path.join(EMBEDDING_DIR, "mul_embeddings_contrastive.npy")
    ],
    "faiss_ready": [
        os.path.join(FAISS_DIR, "pan_index.bin"),
        os.path.join(FAISS_DIR, "mul_index.bin"),
        os.path.join(FAISS_DIR, "pan_index_contrastive.bin"),
        os.path.join(FAISS_DIR, "mul_index_contrastive.bin")
    ],
    "metrics_ready": [
        os.path.join(LOG_DIR, "metrics_summary_baseline.json"),
        os.path.join(LOG_DIR, "metrics_summary_contrastive.json")
    ],
    "retrieval_complete": [
        os.path.join(OUTPUT_DIR, "retrieval_results.png")
    ],
    "visualizations_complete": [
        os.path.join(OUTPUT_DIR, "embeddings_tsne_before.png"),
        os.path.join(OUTPUT_DIR, "embeddings_tsne_after.png")
    ]
}

# Define Stage Dependencies
STAGE_DEPENDENCIES = {
    "dataset_ready": [],
    "baseline_embeddings": ["dataset_ready"],
    "training_complete": ["baseline_embeddings"],
    "contrastive_embeddings": ["training_complete"],
    "faiss_ready": ["baseline_embeddings", "contrastive_embeddings"],
    "metrics_ready": ["faiss_ready"],
    "retrieval_complete": ["faiss_ready"],
    "visualizations_complete": ["baseline_embeddings", "contrastive_embeddings"]
}

def get_pipeline_manifest():
    """
    Loads pipeline_state.json from the active experiment directory.
    If it does not exist, initializes a new one.
    """
    import json
    import time
    manifest_file = os.path.join(get_experiment_dir(), "pipeline_state.json")
    
    default_manifest = {
        "meta": {
            "dataset_hash": get_dataset_hash(),
            "dataset_size": 5000,
            "preprocessing_version": "v1",
            "backbone": BACKBONE,
            "feature_dimension": get_backbone_feature_dim(BACKBONE),
            "projection_dimension": 128,
            "faiss_index_type": FAISS_INDEX,
            "freeze_backbone": FREEZE_BACKBONE,
            "use_full_dataset": USE_FULL_DATASET,
            "experiment_version": "v1"
        },
        "stages": {
            "dataset_ready": False,
            "baseline_embeddings": False,
            "training_complete": False,
            "contrastive_embeddings": False,
            "faiss_ready": False,
            "metrics_ready": False,
            "retrieval_complete": False,
            "visualizations_complete": False
        },
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if not os.path.exists(manifest_file):
        return default_manifest
        
    for attempt in range(3):
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if "meta" not in manifest or "stages" not in manifest:
                return default_manifest
            return manifest
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.1)
        except Exception:
            return default_manifest
            
    return default_manifest

def save_pipeline_manifest(manifest):
    """
    Saves pipeline_state.json atomically using a temp file.
    """
    import json
    import tempfile
    manifest_dir = get_experiment_dir()
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_file = os.path.join(manifest_dir, "pipeline_state.json")
    manifest["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write atomically
    tmp_fd, tmp_path = tempfile.mkstemp(dir=manifest_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=4)
        os.replace(tmp_path, manifest_file)
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        print(f"Warning: Failed to save pipeline manifest: {e}")

def verify_cache(stage_name):
    """
    Verifies if a stage is validly cached.
    Checks:
      1. All required output files exist on disk.
      2. Metadata matches current configuration.
      3. Stage status is true in manifest.
    If 1 and 2 pass but 3 fails, returns (True, "Manifest synchronization issue: outputs exist but manifest not updated").
    If 1 or 2 fail, repairs manifest and returns (False, reason).
    """
    if stage_name == "dataset_ready":
        if os.path.exists(DATASET_PATH):
            return True, None
        return False, f"Dataset file missing at {DATASET_PATH}"
        
    # 1. First validate file existence
    required_files = STAGE_OUTPUTS.get(stage_name, [])
    files_to_check = list(required_files)
        
    if stage_name == "training_complete":
        checkpoint_found = False
        if os.path.exists(CHECKPOINT_DIR):
            for file_name in os.listdir(CHECKPOINT_DIR):
                if file_name.startswith("checkpoint_epoch_") and file_name.endswith(".pth"):
                    checkpoint_found = True
                    break
        if not checkpoint_found:
            invalidate_stage_and_dependents("training_complete", "No checkpoint file found")
            return False, "Missing checkpoint files"

    for f in files_to_check:
        if not os.path.exists(f):
            invalidate_stage_and_dependents(stage_name, f"Missing output file: {os.path.basename(f)}")
            return False, f"Missing output file: {os.path.basename(f)}"

    # 2. Next validate metadata compatibility
    manifest = get_pipeline_manifest()
    meta = manifest.get("meta", {})
    
    feature_dimension = get_backbone_feature_dim(BACKBONE)
    try:
        pan_emb_file = os.path.join(EMBEDDING_DIR, "pan_embeddings.npy")
        if os.path.exists(pan_emb_file):
            import numpy as np
            pan_embs = np.load(pan_emb_file, mmap_mode='r')
            feature_dimension = int(pan_embs.shape[1])
    except Exception:
        pass
        
    dataset_size = 5000
    try:
        labels_file = os.path.join(EMBEDDING_DIR, "labels.npy")
        if os.path.exists(labels_file):
            import numpy as np
            lbls = np.load(labels_file, mmap_mode='r')
            dataset_size = int(lbls.shape[0])
    except Exception:
        pass

    curr_meta = {
        "dataset_hash": get_dataset_hash(),
        "dataset_size": dataset_size,
        "preprocessing_version": "v1",
        "backbone": BACKBONE,
        "feature_dimension": feature_dimension,
        "projection_dimension": 128,
        "faiss_index_type": FAISS_INDEX,
        "freeze_backbone": FREEZE_BACKBONE,
        "use_full_dataset": USE_FULL_DATASET,
        "experiment_version": "v1"
    }
    
    for k, v in curr_meta.items():
        if k == "dataset_size" and not manifest["stages"].get("baseline_embeddings", False):
            continue
        if meta.get(k) != v:
            invalidate_stage_and_dependents(stage_name, f"Metadata mismatch on '{k}': expected {v}, got {meta.get(k)}")
            return False, f"Metadata mismatch on '{k}'"

    # 3. Check manifest status
    if not manifest["stages"].get(stage_name, False):
        return True, "Manifest synchronization issue: outputs exist but manifest not updated"
            
    return True, None

def invalidate_stage_and_dependents(stage_name, reason=None):
    """
    Invalidates a stage and all subsequent dependent stages.
    """
    manifest = get_pipeline_manifest()
    invalidated = []
    
    queue = [stage_name]
    while queue:
        curr = queue.pop(0)
        if manifest["stages"].get(curr, False):
            manifest["stages"][curr] = False
            invalidated.append(curr)
            for dep_stage, prereqs in STAGE_DEPENDENCIES.items():
                if curr in prereqs:
                    queue.append(dep_stage)
                    
    if invalidated:
        if reason:
            print(f"⚠ Cache Inconsistent. Invalidation triggered: {reason}")
            print(f"  Invalidated stages: {', '.join(invalidated)}")
        if not READ_ONLY_MODE:
            save_pipeline_manifest(manifest)

def update_pipeline_manifest(stage_name, status=True):
    """
    Updates completion status for a stage and refreshes metadata.
    """
    manifest = get_pipeline_manifest()
    manifest["stages"][stage_name] = status
    
    feature_dimension = get_backbone_feature_dim(BACKBONE)
    try:
        pan_emb_file = os.path.join(EMBEDDING_DIR, "pan_embeddings.npy")
        if os.path.exists(pan_emb_file):
            import numpy as np
            pan_embs = np.load(pan_emb_file, mmap_mode='r')
            feature_dimension = int(pan_embs.shape[1])
    except Exception:
        pass
        
    dataset_size = 5000
    try:
        labels_file = os.path.join(EMBEDDING_DIR, "labels.npy")
        if os.path.exists(labels_file):
            import numpy as np
            lbls = np.load(labels_file, mmap_mode='r')
            dataset_size = int(lbls.shape[0])
    except Exception:
        pass

    manifest["meta"] = {
        "dataset_hash": get_dataset_hash(),
        "dataset_size": dataset_size,
        "preprocessing_version": "v1",
        "backbone": BACKBONE,
        "feature_dimension": feature_dimension,
        "projection_dimension": 128,
        "faiss_index_type": FAISS_INDEX,
        "freeze_backbone": FREEZE_BACKBONE,
        "use_full_dataset": USE_FULL_DATASET,
        "experiment_version": "v1"
    }
    save_pipeline_manifest(manifest)

def save_model_metadata(model_path):
    """
    Saves a companion metadata file for the saved model checkpoint.
    """
    import json
    import numpy as np
    
    metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
    
    dataset_size = 5000
    try:
        labels_path = os.path.join(EMBEDDING_DIR, "labels.npy")
        if os.path.exists(labels_path):
            labels = np.load(labels_path, mmap_mode='r')
            dataset_size = int(labels.shape[0])
    except Exception:
        pass
        
    feature_dimension = get_backbone_feature_dim(BACKBONE)
    try:
        pan_path = os.path.join(EMBEDDING_DIR, "pan_embeddings.npy")
        if os.path.exists(pan_path):
            pan_embs = np.load(pan_path, mmap_mode='r')
            feature_dimension = int(pan_embs.shape[1])
    except Exception:
        pass

    metadata = {
        "backbone": BACKBONE,
        "feature_dimension": feature_dimension,
        "projection_dimension": 128,
        "faiss_index": FAISS_INDEX,
        "freeze_backbone": FREEZE_BACKBONE,
        "dataset_hash": get_dataset_hash(),
        "preprocessing_version": "v1",
        "experiment_version": "v1",
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    }
    
    try:
        # Write atomically
        import tempfile
        metadata_dir = os.path.dirname(model_path)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=metadata_dir, suffix=".tmp")
        with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        os.replace(tmp_path, metadata_path)
        print(f"Saved model companion metadata to '{metadata_path}'.")
    except Exception as e:
        print(f"Warning: Failed to save model metadata: {e}")

def verify_model_metadata(model_path, expected_backbone=None, expected_feature_dimension=None, expected_projection_dimension=None, expected_dataset_hash=None, strict_hyperparams=False):
    """
    Loads the model companion metadata and verifies compatibility with current configuration.
    If expected_* parameters are None, falls back to the current configuration constants.
    If incompatible, raises a RuntimeError.
    """
    import json
    import numpy as np
    
    metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
    if not os.path.exists(metadata_path):
        print(f"Warning: Model companion metadata file not found at '{metadata_path}'. Skipping validation.")
        return True
        
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read model metadata file at '{metadata_path}': {e}")
        
    # Set default expected values if not provided
    if expected_backbone is None:
        expected_backbone = BACKBONE
        
    if expected_feature_dimension is None:
        expected_feature_dimension = get_backbone_feature_dim(expected_backbone)
        # Try to read feature dimension from saved embeddings if available
        try:
            pan_emb_file = os.path.join(EMBEDDING_DIR, "pan_embeddings.npy")
            if os.path.exists(pan_emb_file):
                pan_embs = np.load(pan_emb_file, mmap_mode='r')
                expected_feature_dimension = int(pan_embs.shape[1])
        except Exception:
            pass
            
    if expected_projection_dimension is None:
        expected_projection_dimension = 128
            
    if expected_dataset_hash is None:
        expected_dataset_hash = get_dataset_hash()

    # Fields to verify
    critical_checks = {
        "backbone": expected_backbone,
        "feature_dimension": expected_feature_dimension,
        "projection_dimension": expected_projection_dimension,
        "dataset_hash": expected_dataset_hash
    }
    
    strict_checks = {
        "faiss_index": FAISS_INDEX,
        "freeze_backbone": FREEZE_BACKBONE,
        "preprocessing_version": "v1",
        "experiment_version": "v1"
    }
    
    checks_to_run = dict(critical_checks)
    if strict_hyperparams:
        checks_to_run.update(strict_checks)
        
    mismatches = []
    for key, expected_val in checks_to_run.items():
        actual_val = metadata.get(key)
        if actual_val != expected_val:
            mismatches.append(f"  - {key}: expected {expected_val}, got {actual_val}")
            
    if mismatches:
        mismatch_str = "\n".join(mismatches)
        error_msg = (
            f"🛑 CONFIGURATION MISMATCH DETECTED!\n"
            f"Model file: {os.path.basename(model_path)}\n"
            f"The following parameters differ from the model's metadata:\n{mismatch_str}\n"
            f"Please ensure your parameters are correct or train a new model."
        )
        raise RuntimeError(error_msg)
        
    print(f"✔ Model companion metadata validation passed for '{model_path}'.")
    return True

def check_stage_skip(stage_name, extra_config=None):
    stage_map = {
        "extract_embeddings": "baseline_embeddings",
        "build_faiss": "faiss_ready",
        "evaluate_contrastive": "contrastive_embeddings",
        "precompute_metrics": "metrics_ready",
        "retrieve": "retrieval_complete",
        "visualize_embeddings": "visualizations_complete"
    }
    mapped_stage = stage_map.get(stage_name, stage_name)
    is_valid, _ = verify_cache(mapped_stage)
    return is_valid

def save_stage_state(stage_name, output_files, extra_config=None):
    stage_map = {
        "extract_embeddings": "baseline_embeddings",
        "build_faiss": "faiss_ready",
        "evaluate_contrastive": "contrastive_embeddings",
        "precompute_metrics": "metrics_ready",
        "retrieve": "retrieval_complete",
        "visualize_embeddings": "visualizations_complete"
    }
    mapped_stage = stage_map.get(stage_name, stage_name)
    update_pipeline_manifest(mapped_stage, True)

def create_faiss_index(embeddings, index_type="auto", metric_type=None):
    """
    Creates a FAISS index based on the chosen index_type and dataset size.
    Returns:
        index: built and trained FAISS index
        resolved_type (str): final resolved index name
        build_time (float): time taken to construct index (seconds)
    """
    import faiss
    import numpy as np
    import time
    
    embeddings = embeddings.astype('float32')
    n, dimension = embeddings.shape
    
    if metric_type is None:
        metric_type = faiss.METRIC_L2
        
    resolved_type = index_type
    if index_type == "auto":
        # Strategy:
        # size < 100,000 -> Flat exact index
        # size < 1,000,000 -> HNSW index
        # size >= 1,000,000 -> IVF PQ index
        if n < 100000:
            resolved_type = "FlatIP" if metric_type == faiss.METRIC_INNER_PRODUCT else "FlatL2"
        elif n < 1000000:
            resolved_type = "HNSW"
        else:
            resolved_type = "IVF PQ"
            
    print(f"Building FAISS index: {resolved_type} | Metric: {'InnerProduct/Cosine' if metric_type == faiss.METRIC_INNER_PRODUCT else 'L2'} | Size: {n} vectors")
    
    start_time = time.time()
    
    if resolved_type == "FlatL2":
        index = faiss.IndexFlatL2(dimension)
    elif resolved_type == "FlatIP":
        index = faiss.IndexFlatIP(dimension)
    elif resolved_type == "HNSW":
        # HNSW Flat with M=32 connections
        index = faiss.IndexHNSWFlat(dimension, 32, metric_type)
    elif resolved_type == "IVF Flat":
        nlist = int(4 * np.sqrt(n))
        nlist = max(1, min(nlist, n // 39))
        quantizer = faiss.IndexFlatL2(dimension) if metric_type == faiss.METRIC_L2 else faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric_type)
    elif resolved_type in ["IVF PQ", "IVF_PQ"]:
        nlist = int(4 * np.sqrt(n))
        nlist = max(1, min(nlist, n // 39))
        # Find divisor for subquantizers
        m = 8
        for possible_m in [32, 16, 8, 4, 2, 1]:
            if dimension % possible_m == 0:
                m = possible_m
                break
        nbits = 8
        quantizer = faiss.IndexFlatL2(dimension) if metric_type == faiss.METRIC_L2 else faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits, metric_type)
    else:
        raise ValueError(f"Unknown FAISS index type: {resolved_type}")
        
    if not index.is_trained:
        print(f"Training FAISS index...")
        index.train(embeddings)
        
    print(f"Adding embeddings to FAISS index...")
    index.add(embeddings)
    build_time = time.time() - start_time
    print(f"FAISS index built in {build_time:.3f} seconds.")
    
    return index, resolved_type, build_time

def save_experiment_metadata(resolved_faiss_index=None):
    """
    Saves or updates the experiment.json metadata file in the experiment directory.
    """
    import json
    import datetime
    import numpy as np
    
    exp_dir = get_experiment_dir()
    os.makedirs(exp_dir, exist_ok=True)
    metadata_file = os.path.join(exp_dir, "experiment.json")
    
    metadata = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        except Exception:
            pass
            
    dataset_size = 5000
    try:
        labels_path = os.path.join(EMBEDDING_DIR, "labels.npy")
        if os.path.exists(labels_path):
            labels = np.load(labels_path, mmap_mode='r')
            dataset_size = int(labels.shape[0])
    except Exception:
        pass
        
    feature_dimension = get_backbone_feature_dim(BACKBONE)
    try:
        pan_path = os.path.join(EMBEDDING_DIR, "pan_embeddings.npy")
        if os.path.exists(pan_path):
            pan_embs = np.load(pan_path, mmap_mode='r')
            feature_dimension = int(pan_embs.shape[1])
    except Exception:
        pass

    metadata.update({
        "dataset": "DSRSID",
        "dataset_size": dataset_size,
        "backbone": BACKBONE,
        "feature_dimension": feature_dimension,
        "projection_dimension": 128,
        "freeze_backbone": FREEZE_BACKBONE,
        "faiss": resolved_faiss_index if resolved_faiss_index is not None else metadata.get("faiss", FAISS_INDEX),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d")
    })
    
    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved experiment metadata to '{metadata_file}'.")
    except Exception as e:
        print(f"Warning: Failed to save experiment metadata: {e}")