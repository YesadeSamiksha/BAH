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

# Setup Roots
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if IS_COLAB:
    STORAGE_ROOT = "/content/drive/MyDrive/BAH"
else:
    STORAGE_ROOT = PROJECT_ROOT

# Centralized Directories
MODEL_DIR = os.path.join(STORAGE_ROOT, "models")
CHECKPOINT_DIR = os.path.join(STORAGE_ROOT, "checkpoints")
EMBEDDING_DIR = os.path.join(STORAGE_ROOT, "embeddings")
FAISS_DIR = os.path.join(STORAGE_ROOT, "faiss_indices")
LOG_DIR = os.path.join(STORAGE_ROOT, "logs")
OUTPUT_DIR = os.path.join(STORAGE_ROOT, "outputs")

# Automatically create missing directories
for directory in [
    MODEL_DIR,
    CHECKPOINT_DIR,
    EMBEDDING_DIR,
    FAISS_DIR,
    LOG_DIR,
    OUTPUT_DIR
]:
    os.makedirs(directory, exist_ok=True)

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

# Centralized Constants / Hyperparameters
FREEZE_BACKBONE = True
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
TEMPERATURE = 0.07
PATIENCE = 5
AUTO_RESUME = True
USE_FULL_DATASET = False

# Versioning Configuration
VERSION = "v1"

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