import os
import sys
import shutil
import datetime

# Detect if running on Google Colab
IS_COLAB = 'google.colab' in sys.modules or 'google.colab' in os.environ
if not IS_COLAB:
    try:
        import google.colab
        IS_COLAB = True
    except ImportError:
        IS_COLAB = False

# Setup Roots
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if IS_COLAB:
    STORAGE_ROOT = "/content/drive/MyDrive/ISRO_Hackathon"
else:
    STORAGE_ROOT = PROJECT_ROOT

# Centralized Directories
DATASET_PATH = os.path.join(STORAGE_ROOT, "data", "DSRSID.mat")
MODEL_DIR = os.path.join(STORAGE_ROOT, "models")
CHECKPOINT_DIR = os.path.join(STORAGE_ROOT, "checkpoints")
EMBEDDING_DIR = os.path.join(STORAGE_ROOT, "embeddings")
FAISS_DIR = os.path.join(STORAGE_ROOT, "faiss_indices")
LOG_DIR = os.path.join(STORAGE_ROOT, "logs")
OUTPUT_DIR = os.path.join(STORAGE_ROOT, "outputs")

# Automatically create missing directories
for directory in [os.path.dirname(DATASET_PATH), MODEL_DIR, CHECKPOINT_DIR, EMBEDDING_DIR, FAISS_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

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