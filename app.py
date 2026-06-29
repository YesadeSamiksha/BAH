import os
import time
import json
import csv
import io
import streamlit as st
import graphviz
import numpy as np
import pandas as pd
import torch
import faiss
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

from dataset import DSRSIDDataset
from train_contrastive import ContrastiveModel, get_backbone_model
from config import DATASET_PATH, STORAGE_ROOT, FREEZE_BACKBONE
import config

@st.cache_resource
def get_dataset(exp_name):
    """
    Loads and caches the dataset handle using @st.cache_resource.
    This ensures h5py reads from the file handle only once and reuses it.
    """
    try:
        paths = get_exp_paths(exp_name)
        emb_dir = paths["EMBEDDING_DIR"]
        indices_path = os.path.join(emb_dir, "subset_indices.npy")
        if not os.path.exists(indices_path):
            return None
        subset_indices = np.load(indices_path)
        return DSRSIDDataset(file_path=DATASET_PATH, indices=subset_indices)
    except Exception as e:
        import logging
        logging.warning(f"Failed to initialize cached dataset handle for {exp_name}: {e}")
        return None

def load_local_image_as_array(path):
    """
    Loads a local image file (like t-SNE png plots) as a NumPy array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image path not found: {path}")
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))

def load_dataset_image(index, modality):
    """
    Validates index and modality, loads the image dynamically from the cached dataset,
    converts PAN to grayscale and MUL to RGB visualization, normalizes pixel values,
    validates the output array shape generally, and returns a NumPy array.
    Tracks rendering stats in session state.
    """
    t0 = time.perf_counter()
    modality_upper = modality.upper()
    try:
        global selected_exp
        if "selected_exp" not in globals() or not selected_exp:
            raise ValueError("Global selected_exp not yet initialized.")
            
        dataset = get_dataset(selected_exp)
        if dataset is None:
            raise ValueError("Dataset handle is not initialized/available.")
            
        # Verify retrieved index is within dataset bounds
        if index < 0 or index >= len(dataset):
            raise IndexError(f"Index {index} is out of bounds for dataset of length {len(dataset)}")
            
        # Verify modality exists
        if modality_upper not in ["PAN", "MUL"]:
            raise ValueError(f"Invalid modality: {modality}. Expected 'PAN' or 'MUL'.")
            
        pan_pil_resized, mul_pil_resized, _ = dataset.get_visualization_images(index)
        
        if modality_upper == "PAN":
            grayscale_pil = pan_pil_resized.convert("L")
            arr = np.array(grayscale_pil)
        else:
            arr = np.array(mul_pil_resized)
            
        # Normalize pixel values
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0).astype(np.uint8)
            
        # General shape validation (not hardcoded to specific resolutions)
        if modality_upper == "PAN":
            # PAN: a 2D grayscale image
            if arr.ndim != 2:
                raise ValueError(f"Invalid PAN image shape: {arr.shape}. Expected a 2D grayscale array.")
        else:
            # MUL: a 3-channel RGB image after visualization
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Invalid MUL image shape: {arr.shape}. Expected a 3-channel (RGB) array.")
                
        # Update debug stats
        if "debug_stats" in st.session_state:
            st.session_state.debug_stats["images_loaded"] += 1
            st.session_state.debug_stats["load_times"].append((time.perf_counter() - t0) * 1000.0)
            
        return arr
    except Exception as e:
        import logging
        logging.warning(f"Error in load_dataset_image for index {index}, modality {modality}: {e}")
        
        # Update debug stats with failure
        if "debug_stats" in st.session_state:
            st.session_state.debug_stats["failed_loads"] += 1
            st.session_state.debug_stats["placeholders_shown"] += 1
            st.session_state.debug_stats["load_times"].append((time.perf_counter() - t0) * 1000.0)
            
        # Safe fallback placeholder image
        if modality_upper == "MUL":
            placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
            placeholder[:, :] = [30, 41, 59] # sleek dark slate blue
        else:
            placeholder = np.zeros((224, 224), dtype=np.uint8)
            placeholder[:, :] = 30 # dark gray
        return placeholder

def safe_display_image(image_input, caption=None):
    """
    Displays a NumPy array safely in Streamlit.
    Accepts NumPy arrays only. Ensures displayed images are freshly generated NumPy arrays 
    for each rerun, preventing stale MediaFileHandler/MediaFileManager warnings.
    """
    try:
        if not isinstance(image_input, np.ndarray):
            raise TypeError(f"safe_display_image accepts NumPy arrays only. Got: {type(image_input)}")
            
        # Make a copy to detach reference and ensure fresh rendering
        arr = image_input.copy()
        
        # Fallback chain for image sizing in different Streamlit versions
        try:
            # Streamlit 1.58.0+ stretch parameter
            st.image(arr, width="stretch", caption=caption, clamp=True)
        except (TypeError, ValueError):
            try:
                # Streamlit 1.30.0 - 1.57.0 parameter
                st.image(arr, use_container_width=True, caption=caption, clamp=True)
            except TypeError:
                # Older Streamlit versions
                st.image(arr, use_column_width=True, caption=caption, clamp=True)
            
    except Exception as e:
        import logging
        logging.warning(f"Error in safe_display_image: {e}")
        try:
            placeholder_arr = np.zeros((224, 224, 3), dtype=np.uint8)
            placeholder_arr[:, :] = [30, 41, 59]
            try:
                st.image(placeholder_arr, width="stretch", caption="Display Failed")
            except (TypeError, ValueError):
                try:
                    st.image(placeholder_arr, use_container_width=True, caption="Display Failed")
                except TypeError:
                    st.image(placeholder_arr, use_column_width=True, caption="Display Failed")
            st.caption(f"Error details: {str(e)}")
        except Exception:
            st.warning("Image display unavailable")

# Page Configuration
st.set_page_config(
    page_title="Cross-Modal Satellite Image Retrieval System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Reset debug stats for the current rerun
st.session_state.debug_stats = {
    "images_loaded": 0,
    "failed_loads": 0,
    "placeholders_shown": 0,
    "load_times": []
}

# Premium Custom CSS
st.markdown("""
<style>
    /* Main Layout */
    .reportview-container {
        background: #0E1117;
        color: #F1F5F9; /* Text Primary */
    }
    
    /* Header styling */
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 2.8rem;
        background: linear-gradient(90deg, #3882F6, #06B6D4, #22C55E, #F59E0B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #F1F5F9;
        font-weight: 600;
    }

    /* Cards */
    .metric-card {
        background-color: #111827; /* Card BG */
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #1F2937; /* Border */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 15px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3882F6; /* Primary Blue */
    }

    /* Result Card Styles */
    .result-card {
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #111827; /* Card BG */
        border: 1.5px solid #1F2937; /* Border */
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .result-card:hover {
        transform: scale(1.01);
        border-color: #3882F6; /* Primary Blue */
    }

    /* Subtext */
    .result-meta {
        font-size: 0.85rem;
        color: #94A3B8; /* Text Secondary */
        margin-top: 5px;
        margin-bottom: 5px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    
    /* Stats box */
    .stats-box {
        background: #111827; /* Card BG */
        border-left: 4px solid #3882F6; /* Primary Blue */
        border-top: 1px solid #1F2937;
        border-right: 1px solid #1F2937;
        border-bottom: 1px solid #1F2937;
        padding: 12px;
        margin-bottom: 15px;
        border-radius: 6px;
        font-family: monospace;
    }
    
    /* Flexbox metrics container */
    .metrics-flex {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around;
        gap: 15px;
        width: 100%;
        padding: 15px;
        background-color: #111827;
        border-radius: 8px;
        border: 1px solid #1F2937;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .metric-flex-item {
        flex: 1 1 15%;
        text-align: center;
        min-width: 90px;
    }
    .metric-flex-val {
        font-size: 1.3rem;
        font-weight: bold;
        color: #F1F5F9;
        font-family: monospace;
    }
    .metric-flex-lbl {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Define Classes
CLASSES = {
    1.0: "Aquafarm",
    2.0: "Cloud",
    3.0: "Forest",
    4.0: "High Building",
    5.0: "Low Building",
    6.0: "Farmland",
    7.0: "River",
    8.0: "Water"
}

def list_available_experiments():
    exp_root = os.path.join(STORAGE_ROOT, "experiments")
    if not os.path.exists(exp_root):
        return []
    exps = []
    for name in os.listdir(exp_root):
        path = os.path.join(exp_root, name)
        if name in ["pipeline_states", "pipeline_state"]:
            continue
        if os.path.isdir(path):
            exps.append(name)
    return exps

def get_default_experiment_name():
    return f"{config.BACKBONE}_{config.FAISS_INDEX.lower()}"

def get_exp_paths(exp_name):
    exp_dir = os.path.join(STORAGE_ROOT, "experiments", exp_name)
    return {
        "MODEL_DIR": os.path.join(exp_dir, "models"),
        "CHECKPOINT_DIR": os.path.join(exp_dir, "checkpoints"),
        "EMBEDDING_DIR": os.path.join(exp_dir, "embeddings"),
        "FAISS_DIR": os.path.join(exp_dir, "faiss_indices"),
        "LOG_DIR": os.path.join(exp_dir, "logs"),
        "OUTPUT_DIR": os.path.join(exp_dir, "outputs")
    }

# Load cached data and models dynamically
@st.cache_resource
def load_resources(exp_name, retrieval_mode):
    paths = get_exp_paths(exp_name)
    emb_dir = paths["EMBEDDING_DIR"]
    faiss_dir = paths["FAISS_DIR"]
    model_dir = paths["MODEL_DIR"]

    indices_path = os.path.join(emb_dir, "subset_indices.npy")
    if not os.path.exists(indices_path):
        st.error(f"Missing '{indices_path}'! Please run extraction for run '{exp_name}' first.")
        st.stop()
        
    subset_indices = np.load(indices_path)
    dataset = get_dataset(exp_name)
    
    labels_file = os.path.join(emb_dir, "labels.npy")
    if not os.path.exists(labels_file):
        st.error(f"Missing '{labels_file}'! Please run extraction for run '{exp_name}' first.")
        st.stop()
    labels = np.load(labels_file)
    
    # Extract backbone name from experiment folder name (e.g. resnet50_hnsw -> resnet50)
    backbone_name = exp_name.split("_")[0]
    
    if retrieval_mode == "baseline":
        pan_embs = np.load(os.path.join(emb_dir, "pan_embeddings.npy"), mmap_mode='r')
        mul_embs = np.load(os.path.join(emb_dir, "mul_embeddings.npy"), mmap_mode='r')
        pan_index = faiss.read_index(os.path.join(faiss_dir, "pan_index.bin"))
        mul_index = faiss.read_index(os.path.join(faiss_dir, "mul_index.bin"))
        
        # Load baseline model dynamically
        model, feature_dim = get_backbone_model(backbone_name)
        model.eval()
    elif retrieval_mode == "contrastive":
        pan_embs = np.load(os.path.join(emb_dir, "pan_embeddings_contrastive.npy"), mmap_mode='r')
        mul_embs = np.load(os.path.join(emb_dir, "mul_embeddings_contrastive.npy"), mmap_mode='r')
        pan_index = faiss.read_index(os.path.join(faiss_dir, "pan_index_contrastive.bin"))
        mul_index = faiss.read_index(os.path.join(faiss_dir, "mul_index_contrastive.bin"))
        
        # Dynamically instantiate ContrastiveModel using target backbone
        # We override config.BACKBONE inside the context to avoid circular issues
        old_backbone = config.BACKBONE
        config.BACKBONE = backbone_name
        model = ContrastiveModel(freeze_backbone=FREEZE_BACKBONE)
        config.BACKBONE = old_backbone
        
        best_model_path = os.path.join(model_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            try:
                expected_dim = pan_embs.shape[1]  # 128 (projection dimension)
                from config import verify_model_metadata, get_backbone_feature_dim
                expected_feat_dim = get_backbone_feature_dim(backbone_name)
                verify_model_metadata(
                    best_model_path,
                    expected_backbone=backbone_name,
                    expected_feature_dimension=expected_feat_dim,
                    expected_projection_dimension=expected_dim,
                    expected_dataset_hash=config.get_dataset_hash(),
                    strict_hyperparams=False
                )
            except RuntimeError as e:
                st.error(f" Model Incompatibility Detected:\n\n{str(e)}")
                st.stop()
            model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
        model.eval()
    else:
        raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")

    return {
        "dataset": dataset,
        "labels": labels,
        "pan_embs": pan_embs,
        "mul_embs": mul_embs,
        "pan_index": pan_index,
        "mul_index": mul_index,
        "model": model,
        "feature_dim": pan_embs.shape[1],
        "backbone_name": backbone_name
    }

@st.cache_resource
def load_metrics_summaries(exp_name):
    paths = get_exp_paths(exp_name)
    log_dir = paths["LOG_DIR"]
    
    baseline_file = os.path.join(log_dir, "metrics_summary_baseline.json")
    contrastive_file = os.path.join(log_dir, "metrics_summary_contrastive.json")
    
    baseline_metrics = {}
    if os.path.exists(baseline_file):
        with open(baseline_file, "r") as f:
            baseline_metrics = json.load(f)
            
    contrastive_metrics = {}
    if os.path.exists(contrastive_file):
        with open(contrastive_file, "r") as f:
            contrastive_metrics = json.load(f)
            
    return {
        "baseline": baseline_metrics,
        "contrastive": contrastive_metrics
    }

# Streamlit Title
st.title("Cross-Modal Satellite Image Retrieval System")
st.markdown("##### *Aligning Panchromatic & Multispectral Satellite Data with Supervised Contrastive Learning*")

# Discover and load experiments
available_exps = list_available_experiments()
default_name = get_default_experiment_name()
if default_name not in available_exps:
    available_exps.append(default_name)

# Sidebar Controls
st.sidebar.header(" Experiment Run Configuration")
selected_exp = st.sidebar.selectbox(
    "Select Experiment Run",
    options=available_exps,
    index=available_exps.index(default_name) if default_name in available_exps else 0,
    help="Select which experiment configuration (backbone and FAISS index type) to load."
)

paths = get_exp_paths(selected_exp)

# Sidebar - Retrieval Mode configuration
retrieval_model_option = st.sidebar.radio(
    "Retrieval Model",
    options=["Baseline", "Contrastive"],
    index=1,
    help="Switch between Baseline and Contrastive model representations."
)
active_mode = retrieval_model_option.lower()

res = load_resources(selected_exp, active_mode)
metrics_summaries = load_metrics_summaries(selected_exp)

retrieval_mode = st.sidebar.selectbox(
    "Select Retrieval Mode",
    ["PAN → MUL", "MUL → PAN", "PAN → PAN", "MUL → MUL"],
    index=0,
    help="PAN -> MUL queries Panchromatic and retrieves Multispectral (cross-modal), and vice versa."
)

query_source = st.sidebar.radio(
    "Query Source Mode",
    ["Dataset Index Mode", "File Upload Mode"],
    index=0
)

# Load experiment metadata if it exists
metadata_file = os.path.join(paths["MODEL_DIR"], "..", "experiment.json")
metadata = {}
if os.path.exists(metadata_file):
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    except Exception:
        pass

# Display Experiment Info in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader(" Active Experiment Details")
if metadata:
    st.sidebar.markdown(f"""
    **Dataset**: `{metadata.get('dataset', 'DSRSID')}` ({metadata.get('dataset_size', 5000):,} samples)  
    **Backbone**: `{metadata.get('backbone', 'resnet18').upper()}`  
    **Feature Dim**: `{metadata.get('feature_dimension', config.get_backbone_feature_dim(metadata.get('backbone', 'resnet18')))}`  
    **Projection Dim**: `{metadata.get('projection_dimension', 128)}`  
    **Index Type**: `{metadata.get('faiss', 'HNSW')}`  
    **Freeze Backbone**: `{metadata.get('freeze_backbone', True)}`  
    **Training Date**: `{metadata.get('timestamp', 'N/A')}`  
    """)
else:
    st.sidebar.info("No experiment.json metadata found for this run.")

# Sidebar - Debug UI toggle
st.sidebar.markdown("---")
st.sidebar.subheader(" Developer / Debug Diagnostics")
DEBUG_UI = st.sidebar.checkbox(
    "Enable UI Debug Mode",
    value=False,
    help="When enabled, displays performance stats, dataset loading times, cache info, and counts."
)

# Demo Mode Shortcut Selector
st.sidebar.markdown("---")
st.sidebar.subheader(" Demo Examples (One-Click)")
cols_demo1, cols_demo2 = st.sidebar.columns(2)

demo_idx = None
num_samples = len(res["labels"])
samples_per_class = num_samples // 8

with cols_demo1:
    if st.button(" Forest Example"):
        demo_idx = int(2.4 * samples_per_class)
    if st.button(" Urban Example"):
        demo_idx = int(3.2 * samples_per_class)
with cols_demo2:
    if st.button(" Water Example"):
        demo_idx = int(7.4 * samples_per_class)
    if st.button(" River Example"):
        demo_idx = int(6.4 * samples_per_class)

# Tabs Layout
tab_retrieval, tab_eval, tab_embeddings, tab_arch, tab_about = st.tabs([
    " Retrieval Sandbox", 
    " Performance Dashboard", 
    " t-SNE Embeddings", 
    " Pipeline Architecture", 
    " About Project"
])

# ----------------- TAB 1: RETRIEVAL SANDBOX -----------------
with tab_retrieval:
    st.subheader("Interactive Query Interface")
    
    if 'query_idx_input' not in st.session_state:
        st.session_state.query_idx_input = 1500

    if demo_idx is not None:
        st.session_state.query_idx_input = demo_idx
        
    query_idx = st.session_state.query_idx_input
    
    query_img_arr = None
    query_label_val = None
    query_embedding = None
    latency_extract = 0.0
    latency_search = 0.0

    if query_source == "Dataset Index Mode":
        st.markdown(f"Select a query index from the {num_samples:,} stratified samples. Images and metadata will load from the DSRSID dataset.")
        
        # Resolve Session State Warning: Do not pass default value parameter when key is set
        query_idx = st.number_input(
            f"Dataset Sample Index (0 - {num_samples - 1})", 
            min_value=0, max_value=num_samples - 1, 
            key='query_idx_input',
            help=f"Select index. The {num_samples} samples are stratified ({samples_per_class} per class: 0-{samples_per_class-1} Class 1, ...)"
        )
        
        # Load query image dynamically on-demand from dataset as a NumPy array (Requirement 1, 3, 4)
        query_modality = "PAN" if retrieval_mode.startswith("PAN") else "MUL"
        query_img_arr = load_dataset_image(query_idx, query_modality)
        query_label_val = float(res["labels"][query_idx])
        
        if retrieval_mode.startswith("PAN"):
            query_embedding = res["pan_embs"][query_idx]
        else:
            query_embedding = res["mul_embs"][query_idx]
            
        latency_extract = 0.0
        
    else:  # File Upload Mode
        st.markdown("Upload a custom satellite image to perform real-time encoder inference and retrieval.")
        
        uploaded_file = st.file_uploader(
            f"Upload {'Panchromatic' if retrieval_mode.startswith('PAN') else 'Multispectral RGB'} Image",
            type=["png", "jpg", "jpeg"]
        )
        
        if uploaded_file is not None:
            # Read uploaded file as PIL and convert to NumPy array (Requirement 1, 3, 5)
            query_pil = Image.open(uploaded_file)
            query_img_arr = np.array(query_pil.convert("RGB"))
            
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            t0 = time.perf_counter()
            img_rgb = query_pil.convert("RGB")
            img_tensor = preprocess(img_rgb).unsqueeze(0)
            
            if active_mode == "baseline":
                with torch.no_grad():
                    emb_torch = res["model"](img_tensor)
            else:
                with torch.no_grad():
                    if retrieval_mode.startswith("PAN"):
                        emb_torch = res["model"].forward_pan(img_tensor)
                    else:
                        emb_torch = res["model"].forward_mul(img_tensor)
                    
            query_embedding = emb_torch.squeeze(0).numpy()
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            t1 = time.perf_counter()
            latency_extract = (t1 - t0) * 1000.0
            query_label_val = None
        else:
            st.info("Please upload an image file to perform retrieval.")
            
    if query_embedding is not None and query_img_arr is not None:
        target_index = res["mul_index"] if retrieval_mode.endswith("MUL") else res["pan_index"]
        target_modality = "MUL" if retrieval_mode.endswith("MUL") else "PAN"
        
        t0 = time.perf_counter()
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        exclude_self_idx = query_idx if query_source == "Dataset Index Mode" else None
        search_k = 6 if exclude_self_idx is not None else 5
        
        similarities, indices = target_index.search(query_vector, search_k)
        
        retrieved_idxs = []
        retrieved_sims = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            if exclude_self_idx is not None and idx == exclude_self_idx:
                continue
            retrieved_idxs.append(int(idx))
            retrieved_sims.append(float(sim))
            if len(retrieved_idxs) == 5:
                break
                
        t1 = time.perf_counter()
        latency_search = (t1 - t0) * 1000.0
        
        # Preload retrieved images to make sure debug stats are accurate before rendering
        ret_imgs = []
        for ret_idx in retrieved_idxs:
            img = load_dataset_image(ret_idx, target_modality)
            ret_imgs.append(img)
            
        col_query, col_results = st.columns([1, 3])
        
        with col_query:
            st.markdown("###  Query Image")
            safe_display_image(query_img_arr)
            
            label_text = CLASSES.get(query_label_val, "Unknown (Uploaded File)")
            
            # Query Information Panel (Requirement 5)
            query_modality = "PAN" if retrieval_mode.startswith("PAN") else "MUL"
            backbone_display = res["backbone_name"].upper()
            st.markdown(f"""
            <div class="metric-card" style="
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
                border: 1px solid rgba(56, 189, 248, 0.2);
                border-left: 5px solid #38BDF8;
                padding: 18px;
                margin-top: 15px;
                border-radius: 12px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
                backdrop-filter: blur(4px);
            ">
                <h4 style="margin: 0 0 12px 0; color: #FFFFFF; font-size: 1.15rem; font-weight: 600; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 8px; font-family: 'Outfit', sans-serif;"> Query Specifications</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; font-family: 'Inter', sans-serif; color: #CBD5E1;">
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 8px 0; color: #94A3B8; font-weight: 500;">Dataset</td>
                        <td style="padding: 8px 0; text-align: right; color: #FFFFFF; font-weight: 600;">DSRSID</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 8px 0; color: #94A3B8; font-weight: 500;">Query Modality</td>
                        <td style="padding: 8px 0; text-align: right; color: #FFFFFF; font-weight: 600;">{query_modality}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 8px 0; color: #94A3B8; font-weight: 500;">Query Class</td>
                        <td style="padding: 8px 0; text-align: right; color: #38BDF8; font-weight: 600;">{label_text}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 8px 0; color: #94A3B8; font-weight: 500;">Encoder Backbone</td>
                        <td style="padding: 8px 0; text-align: right; color: #FFFFFF; font-weight: 600;">{backbone_display}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05);">
                        <td style="padding: 8px 0; color: #94A3B8; font-weight: 500;">Embedding Dim</td>
                        <td style="padding: 8px 0; text-align: right; color: #FFFFFF; font-weight: 600;">{res["feature_dim"]}D</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #94A3B8; font-weight: 500;">Retrieval Mode</td>
                        <td style="padding: 8px 0; text-align: right; color: #F59E0B; font-weight: 600;">{retrieval_mode}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            if DEBUG_UI:
                stats = st.session_state.get("debug_stats", {})
                loaded = stats.get("images_loaded", 0)
                failed = stats.get("failed_loads", 0)
                placeholders = stats.get("placeholders_shown", 0)
                times = stats.get("load_times", [])
                avg_time = np.mean(times) if times else 0.0
                
                st.markdown("#####  UI Debug Stats")
                st.markdown(f"""
                <div class="stats-box" style="
                    background: #111827; 
                    border-left: 4px solid #EF4444; 
                    padding: 12px; 
                    margin-bottom: 15px; 
                    border-radius: 6px; 
                    font-family: 'Space Mono', monospace;
                    font-size: 0.85rem;
                    color: #FCA5A5;
                ">
                Images Loaded  : {loaded}<br>
                Failed Loads   : {failed}<br>
                Placeholders   : {placeholders}<br>
                Avg Load Time  : {avg_time:.2f} ms<br>
                Cache Handle   : Reused (st.cache_resource)
                </div>
                """, unsafe_allow_html=True)
            
        with col_results:
            st.markdown("###  Top-5 Retrieved Matches")
            
            res_cols = st.columns(5)
            csv_data = []
            same_class_count = 0
            diff_class_count = 0
            avg_sim_pct = []
            
            latency_total = latency_extract + latency_search
            
            for r, (ret_idx, similarity) in enumerate(zip(retrieved_idxs, retrieved_sims)):
                try:
                    ret_lbl_val = float(res["labels"][ret_idx])
                    ret_class_name = CLASSES.get(ret_lbl_val, "Unknown")
                    ret_img = ret_imgs[r]
                except Exception as e:
                    st.warning(f"️ Failed to load retrieved match at rank {r+1} (index {ret_idx}): {e}")
                    continue
                
                is_match = (query_label_val is not None and ret_lbl_val == query_label_val)
                
                # Check metric display representation
                if active_mode == "baseline":
                    # L2 Score converted to similarity percentage
                    similarity_pct = max(0.0, min(100.0, (1.0 - (similarity / 400.0)) * 100.0))
                    calc_dist = similarity
                    score_header = "L2_Distance"
                else:
                    # Cosine similarity converted to similarity percentage
                    similarity_pct = max(0.0, min(100.0, similarity * 100.0))
                    calc_dist = 1.0 - similarity
                    score_header = "Similarity_Score"
                    
                avg_sim_pct.append(similarity_pct)
                
                # Determine class correctness badge
                if query_label_val is None:
                    status_text = "? Unknown"
                    status_color = "#F59E0B" # Warning Amber
                    status_bg = "rgba(245, 158, 11, 0.15)"
                    status_border = "#F59E0B"
                    same_class_status = "Unknown"
                elif is_match:
                    status_text = " Same Class"
                    status_color = "#22C55E" # Success Green
                    status_bg = "rgba(34, 197, 94, 0.15)"
                    status_border = "#22C55E"
                    same_class_count += 1
                    same_class_status = "Yes"
                else:
                    status_text = " Different Class"
                    status_color = "#EF4444" # Danger Red
                    status_bg = "rgba(239, 68, 68, 0.15)"
                    status_border = "#EF4444"
                    diff_class_count += 1
                    same_class_status = "No"
                    
                confidence_color = "#22C55E" if similarity_pct >= 90.0 else "#F59E0B" if similarity_pct >= 70.0 else "#EF4444"
                
                # Dynamic interpolation for percentile rank
                if active_mode == "contrastive":
                    if similarity >= 0.998:
                        pct_str = "Top 0.10%"
                    elif similarity >= 0.995:
                        pct_str = f"Top {0.10 + (0.998 - similarity)/(0.998 - 0.995)*0.40:.2f}%"
                    elif similarity >= 0.990:
                        pct_str = f"Top {0.50 + (0.995 - similarity)/(0.995 - 0.990)*0.70:.2f}%"
                    elif similarity >= 0.970:
                        pct_str = f"Top {1.20 + (0.990 - similarity)/(0.990 - 0.970)*3.80:.2f}%"
                    elif similarity >= 0.950:
                        pct_str = f"Top {5.00 + (0.970 - similarity)/(0.970 - 0.950)*5.00:.2f}%"
                    elif similarity >= 0.900:
                        pct_str = f"Top {10.00 + (0.950 - similarity)/(0.950 - 0.900)*5.00:.2f}%"
                    else:
                        pct_str = f"Top {15.00 + min(35.0, (0.90 - similarity)*50.0):.2f}%"
                else:
                    if similarity <= 50.0:
                        pct_str = "Top 0.10%"
                    elif similarity <= 100.0:
                        pct_str = f"Top {0.10 + (similarity - 50.0)/50.0*0.90:.2f}%"
                    elif similarity <= 140.0:
                        pct_str = f"Top {1.00 + (similarity - 100.0)/40.0*4.00:.2f}%"
                    elif similarity <= 150.0:
                        pct_str = f"Top {5.00 + (similarity - 140.0)/10.0*5.00:.2f}%"
                    elif similarity <= 160.0:
                        pct_str = f"Top {10.00 + (similarity - 150.0)/10.0*10.00:.2f}%"
                    else:
                        pct_str = f"Top {20.00 + min(30.0, (similarity - 160.0)*0.5):.2f}%"

                with res_cols[r]:
                    with st.container(border=True):
                        # Rank and correctness status
                        st.markdown(f"**Rank {r+1}** | <span style='color: {status_color}; font-weight: bold;'>{status_text}</span>", unsafe_allow_html=True)
                        
                        # Display retrieved image
                        safe_display_image(ret_img)
                        
                        # Similarity progress bar
                        st.progress(similarity_pct / 100.0)
                        
                        # Factual metadata specifications
                        st.markdown(f"""
                        **{ret_class_name}**  
                        Index: `{ret_idx}`
                        
                        - Dataset: `DSRSID`
                        - Modality: `{target_modality}`
                        - Resolution: `224 × 224`
                        - Distance: `{calc_dist:.4f}`
                        """)
                
                csv_data.append({
                    "Rank": r + 1,
                    "Dataset_Index": ret_idx,
                    score_header: f"{similarity:.5f}",
                    "Class_Label": ret_class_name,
                    "Is_Correct": "Yes" if is_match else "No"
                })
            
            # Bottom section - Interpretation and Retrieval Summary using native Streamlit components
            st.markdown("---")
            col_interpret, col_summary = st.columns([1, 1])
            
            with col_interpret:
                st.subheader("Similarity Interpretation")
                st.write(f"These results are ranked based on **{'cosine similarity' if active_mode == 'contrastive' else 'Euclidean (L2) distance'}** in the learned embedding space.")
                st.write(f"- **{'Higher cosine similarity indicates that the retrieved image is closer to the query in the learned embedding space.' if active_mode == 'contrastive' else 'Lower L2 distance indicates that the retrieved image is closer to the query in the feature space.'}**")
                st.write("- **Percentile** indicates how rare this match is in the entire database.")
                st.write("- **Confidence Guide**: High (>= 90% Similarity), Medium (70% - 90% Similarity), Low (< 70% Similarity).")
                
            with col_summary:
                st.subheader("Retrieval Summary")
                avg_similarity_val = np.mean(avg_sim_pct) if avg_sim_pct else 0.0
                same_class_val = f"{same_class_count} / 5" if query_label_val is not None else "N/A"
                
                st.markdown(f"""
                <div class="metrics-flex">
                    <div class="metric-flex-item">
                        <div class="metric-flex-lbl">Average Similarity</div>
                        <div class="metric-flex-val">{avg_similarity_val:.2f}%</div>
                    </div>
                    <div class="metric-flex-item">
                        <div class="metric-flex-lbl">Same Class Match</div>
                        <div class="metric-flex-val">{same_class_val}</div>
                    </div>
                    <div class="metric-flex-item">
                        <div class="metric-flex-lbl">Best Match Rank</div>
                        <div class="metric-flex-val">Rank 1</div>
                    </div>
                    <div class="metric-flex-item">
                        <div class="metric-flex-lbl">Retrieval Time</div>
                        <div class="metric-flex-val">{latency_total:.2f} ms</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Bottom row - Retrieval Latency and Overall Model Metrics using native Streamlit components
            st.markdown("---")
            st.subheader("Performance & Diagnostics Metrics")
            col_lat_bottom, col_metric_bottom = st.columns([1, 2])
            
            with col_lat_bottom:
                st.markdown("#### Retrieval Latency")
                st.markdown(f"""
                <div class="metrics-flex">
                    <div class="metric-flex-item">
                        <div class="metric-flex-lbl">Inference</div>
                        <div class="metric-flex-val">{latency_extract:.2f} ms</div>
                    </div>
                    <div class="metric-flex-item">
                        <div class="metric-flex-lbl">FAISS Index</div>
                        <div class="metric-flex-val">{latency_search:.2f} ms</div>
                    </div>
                    <div class="metric-flex-item">
                        <div class="metric-flex-lbl">Total Time</div>
                        <div class="metric-flex-val" style="color: #22C55E;">{latency_total:.2f} ms</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_metric_bottom:
                st.markdown("#### Overall Model Metrics (This Run)")
                mode_key = retrieval_mode.replace(" → ", "_")
                active_metrics = metrics_summaries.get(active_mode, {}).get(mode_key, {})
                if active_metrics:
                    p5_val = active_metrics.get('precision_5', 0.0)*100
                    r5_val = active_metrics.get('recall_5', 0.0)*100
                    f5_val = active_metrics.get('f1_5', 0.0)*100
                    map5_val = active_metrics.get('map_5', 0.0)*100
                    map10_val = active_metrics.get('map_10', 0.0)*100
                    
                    st.markdown(f"""
                    <div class="metrics-flex">
                        <div class="metric-flex-item">
                            <div class="metric-flex-lbl">Precision@5</div>
                            <div class="metric-flex-val">{p5_val:.2f}%</div>
                        </div>
                        <div class="metric-flex-item">
                            <div class="metric-flex-lbl">Recall@5</div>
                            <div class="metric-flex-val">{r5_val:.2f}%</div>
                        </div>
                        <div class="metric-flex-item">
                            <div class="metric-flex-lbl">F1-Score@5</div>
                            <div class="metric-flex-val">{f5_val:.2f}%</div>
                        </div>
                        <div class="metric-flex-item">
                            <div class="metric-flex-lbl">mAP@5</div>
                            <div class="metric-flex-val">{map5_val:.2f}%</div>
                        </div>
                        <div class="metric-flex-item">
                            <div class="metric-flex-lbl">mAP@10</div>
                            <div class="metric-flex-val">{map10_val:.2f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Metrics not precomputed.")

            csv_buffer = io.StringIO()
            score_header = "L2_Distance" if active_mode == "baseline" else "Similarity_Score"
            writer = csv.DictWriter(csv_buffer, fieldnames=["Rank", "Dataset_Index", score_header, "Class_Label", "Is_Correct"])
            writer.writeheader()
            writer.writerows(csv_data)
            
            st.markdown("---")
            mode_key = retrieval_mode.replace(" → ", "_")
            st.download_button(
                label=" Download Retrieval Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"retrieval_results_idx_{query_idx}.csv",
                mime="text/csv",
                key=f"download_{query_idx}_{active_mode}_{mode_key}"
            )

# ----------------- TAB 2: PERFORMANCE DASHBOARD -----------------
with tab_eval:
    st.subheader("Model Performance & Evaluation Dashboard")
    st.markdown(f"Comparison between the **Baseline {res['feature_dim']}D (L2)** and the **Supervised Contrastive 128D (Cosine)** models for experiment run `{selected_exp}`.")
    
    ms = metrics_summaries
    
    if not ms or "baseline" not in ms or not ms["baseline"]:
        st.warning("No pre-computed metrics found. Run 'precompute_metrics.py' to generate.")
    else:
        st.markdown("####  Before vs. After Contrastive Learning")
        
        eval_cols = st.columns(4)
        modes_keys = [
            ("PAN → MUL (Cross-Modal)", "PAN_MUL"),
            ("MUL → PAN (Cross-Modal)", "MUL_PAN"),
            ("PAN → PAN (Intra-Modal)", "PAN_PAN"),
            ("MUL → MUL (Intra-Modal)", "MUL_MUL")
        ]
        
        for idx, (title, key) in enumerate(modes_keys):
            with eval_cols[idx]:
                b_p5 = ms["baseline"][key]["precision_5"] * 100
                c_p5 = ms["contrastive"][key]["precision_5"] * 100
                delta_p5 = c_p5 - b_p5
                
                st.markdown(f"""
                <div class="metric-card">
                    <h5 style="margin-top:0px; color:#94A3B8;">{title}</h5>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric(
                    label="Precision@5",
                    value=f"{c_p5:.2f}%",
                    delta=f"+{delta_p5:.2f}%" if delta_p5 > 0 else f"{delta_p5:.2f}%",
                    delta_color="normal"
                )
                
                st.markdown(f"**Baseline**: `{b_p5:.2f}%`  \n**mAP@5**: `{ms['contrastive'][key]['map_5']*100:.2f}%`  \n**mAP@10**: `{ms['contrastive'][key]['map_10']*100:.2f}%`")
                
        st.markdown("---")
        st.markdown("####  Precision@5 Retrieval Mode Comparison")
        
        chart_data = {
            "Retrieval Mode": ["PAN→PAN", "MUL→MUL", "PAN→MUL", "MUL→PAN"],
            f"Baseline ({res['feature_dim']}D L2)": [
                ms["baseline"]["PAN_PAN"]["precision_5"] * 100,
                ms["baseline"]["MUL_MUL"]["precision_5"] * 100,
                ms["baseline"]["PAN_MUL"]["precision_5"] * 100,
                ms["baseline"]["MUL_PAN"]["precision_5"] * 100,
            ],
            "Contrastive (128D Cosine)": [
                ms["contrastive"]["PAN_PAN"]["precision_5"] * 100,
                ms["contrastive"]["MUL_MUL"]["precision_5"] * 100,
                ms["contrastive"]["PAN_MUL"]["precision_5"] * 100,
                ms["contrastive"]["MUL_PAN"]["precision_5"] * 100,
            ]
        }
        
        df_chart = pd.DataFrame(chart_data).set_index("Retrieval Mode")
        st.bar_chart(df_chart, height=350)
        
        # Load local engineering benchmark if available
        benchmark_file = os.path.join(paths["LOG_DIR"], "retrieval_benchmark.json")
        if os.path.exists(benchmark_file):
            st.markdown("---")
            st.markdown("####  Engineering Retrieval Performance Benchmarks")
            with open(benchmark_file, "r") as bf:
                b_data = json.load(bf)
            
            bench_cols = st.columns(4)
            with bench_cols[0]:
                st.metric("Avg Search Latency", f"{b_data.get('avg_search_latency_ms', 0.0):.4f} ms")
            with bench_cols[1]:
                st.metric("Queries Per Second (QPS)", f"{b_data.get('queries_per_second', 0.0):.1f}")
            with bench_cols[2]:
                st.metric("FAISS Index Type", f"{b_data.get('mul_resolved_index_type', 'HNSW')}")
            with bench_cols[3]:
                st.metric("Index Size (MUL)", f"{b_data.get('mul_index_file_size_bytes', 0) / (1024**2):.2f} MB")

        # Dynamic CPU Latency Benchmark
        st.markdown("---")
        st.markdown("####  Real-Time Local CPU Latency Benchmarking")
        
        if st.button(" Run CPU Latency Benchmark (20 Queries)"):
            with st.spinner("Running search benchmarks on CPU..."):
                ext_times = []
                search_times = []
                
                random_idxs = np.random.choice(5000, 20, replace=False)
                
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                dummy_pil = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
                dummy_tensor = preprocess(dummy_pil).unsqueeze(0)
                
                for test_idx in random_idxs:
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        if active_mode == "baseline":
                            _ = res["model"](dummy_tensor)
                        else:
                            _ = res["model"].forward_pan(dummy_tensor)
                    t1 = time.perf_counter()
                    ext_times.append((t1 - t0) * 1000.0)
                    
                    q_emb = res["pan_embs"][test_idx].reshape(1, -1).astype('float32')
                    t0 = time.perf_counter()
                    _, _ = res["mul_index"].search(q_emb, 5)
                    t1 = time.perf_counter()
                    search_times.append((t1 - t0) * 1000.0)
                    
                avg_ext = np.mean(ext_times)
                avg_search = np.mean(search_times)
                
                st.success("Benchmark completed successfully!")
                
                cols_lat = st.columns(2)
                with cols_lat[0]:
                    st.metric("Avg Embedding Extraction Time", f"{avg_ext:.2f} ms", help=f"Time taken by {res['backbone_name'].upper()} + Projection Head forward pass on CPU.")
                with cols_lat[1]:
                    st.metric("Avg FAISS Index Search Time", f"{avg_search:.2f} ms", help="Time taken by FAISS index search.")

# ----------------- TAB 3: T-SNE EMBEDDINGS -----------------
with tab_embeddings:
    st.subheader("t-SNE Embedding Space Projections")
    st.markdown("Visualizing the alignment of 800 sampled PAN (circles `o`) and MUL (triangles `^`) vectors in 2D space before and after contrastive learning.")
    
    col_tsne_before, col_tsne_after = st.columns(2)
    
    with col_tsne_before:
        st.markdown(f"#####  Before Contrastive Training (Baseline {res['feature_dim']}D)")
        before_tsne_path = os.path.join(paths["OUTPUT_DIR"], "embeddings_tsne_before.png")
        if os.path.exists(before_tsne_path):
            try:
                img_arr = load_local_image_as_array(before_tsne_path)
                safe_display_image(img_arr)
            except Exception as e:
                st.error(f"Failed to load image: {e}")
            st.markdown("<p style='font-size:0.85rem; color:#94A3B8; text-align:center;'>Modalities are completely separated. PAN and MUL representations of the same class do not align in the shared space.</p>", unsafe_allow_html=True)
        else:
            st.info("Missing 'embeddings_tsne_before.png' image for this experiment.")
            
    with col_tsne_after:
        st.markdown("#####  After Supervised Contrastive Training (Aligned 128D)")
        after_tsne_path = os.path.join(paths["OUTPUT_DIR"], "embeddings_tsne_after.png")
        if os.path.exists(after_tsne_path):
            try:
                img_arr = load_local_image_as_array(after_tsne_path)
                safe_display_image(img_arr)
            except Exception as e:
                st.error(f"Failed to load image: {e}")
            st.markdown("<p style='font-size:0.85rem; color:#94A3B8; text-align:center;'>PAN and MUL representations of the same classes merge and overlap, showing successful cross-modal alignment.</p>", unsafe_allow_html=True)
        else:
            st.info("Missing 'embeddings_tsne_after.png' image for this experiment.")

# ----------------- TAB 4: ARCHITECTURE DIAGRAM -----------------
with tab_arch:
    st.subheader("Dual-Encoder Embedding Alignment Architecture")
    st.markdown(f"Below is the flow of Panchromatic (PAN) and Multispectral (MUL) modalities through their frozen {res['backbone_name'].upper()} backbones and trainable projection heads into the aligned 128D embedding space.")
    
    st.graphviz_chart(f'''
    digraph G {{
        rankdir=LR;
        node [style=filled, fillcolor="#1E293B", color="#38BDF8", fontcolor="#F8FAFC", fontname="Arial", shape=box, penwidth=1.5];
        edge [color="#38BDF8", fontname="Arial", fontcolor="#94A3B8", penwidth=1.5];
        bgcolor="transparent";
        
        subgraph cluster_pan {{
            label = "Panchromatic (PAN) Stream";
            fontname="Arial";
            color = "#3B82F6";
            style = dashed;
            fontcolor = "#3B82F6";
            
            PAN [label="PAN Image\\n(1 x 256 x 256)", fillcolor="#0F172A"];
            PAN_Enc [label="{res['backbone_name'].upper()} backbone\\n(fc layer replaced with Identity)\\n(Frozen)", shape=ellipse];
            PAN_Proj [label="Projection Head\\nLinear({res['feature_dim']}, 256) -> ReLU -> Linear(256, 128)\\n(Trainable)"];
            
            PAN -> PAN_Enc -> PAN_Proj;
        }}
        
        subgraph cluster_mul {{
            label = "Multispectral (MUL) Stream";
            fontname="Arial";
            color = "#10B981";
            style = dashed;
            fontcolor = "#10B981";
            
            MUL [label="MUL RGB Image\\n(3 x 64 x 64)", fillcolor="#0F172A"];
            MUL_Enc [label="{res['backbone_name'].upper()} backbone\\n(fc layer replaced with Identity)\\n(Frozen)", shape=ellipse];
            MUL_Proj [label="Projection Head\\nLinear({res['feature_dim']}, 256) -> ReLU -> Linear(256, 128)\\n(Trainable)"];
            
            MUL -> MUL_Enc -> MUL_Proj;
        }}
        
        Shared [label="Shared 128D Space\\n(L2 Normalized Vectors)", shape=hexagon, fillcolor="#3F2B3E", color="#EC4899", fontcolor="#F472B6"];
        
        PAN_Proj -> Shared [label="Cosine Similarity\\n(FAISS Search)", constraint=true];
        MUL_Proj -> Shared [label="Cosine Similarity\\n(FAISS Search)", constraint=true];
    }}
    ''')

# ----------------- TAB 5: ABOUT PROJECT -----------------
with tab_about:
    st.subheader("Project Information & Hackathon Details")
    
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin-top:0px; color:#4285F4;"> Dataset Card: DSRSID</h4>
        <p><b>DSRSID</b> is a high-resolution satellite imagery dataset designed for dual-modal remote sensing representation tasks.</p>
        <table style="width:100%; border-collapse: collapse; margin-top: 10px;">
            <tr style="border-bottom: 1px solid #334155;">
                <td style="padding: 8px 0; color:#94A3B8;"><b>Total Samples</b></td>
                <td style="padding: 8px 0;">80,000 Paired Images</td>
            </tr>
            <tr style="border-bottom: 1px solid #334155;">
                <td style="padding: 8px 0; color:#94A3B8;"><b>Panchromatic (PAN) Images</b></td>
                <td style="padding: 8px 0;">80,000 samples @ Shape (1, 256, 256)</td>
            </tr>
            <tr style="border-bottom: 1px solid #334155;">
                <td style="padding: 8px 0; color:#94A3B8;"><b>Multispectral (MUL) Images</b></td>
                <td style="padding: 8px 0;">80,000 samples @ Shape (4, 64, 64) [Bands: R, G, B, NIR]</td>
            </tr>
            <tr style="border-bottom: 1px solid #334155;">
                <td style="padding: 8px 0; color:#94A3B8;"><b>Classes (8 land cover categories)</b></td>
                <td style="padding: 8px 0;">Aquafarm, Cloud, Forest, High Building, Low Building, Farmland, River, Water (10,000 samples each)</td>
            </tr>
            <tr>
                <td style="padding: 8px 0; color:#94A3B8;"><b>Subset for Experimentation</b></td>
                <td style="padding: 8px 0;"><b>5,000 samples</b> (Stratified: 625 samples per class)</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("###  Core Concepts Explained")
    
    st.markdown("""
    #### 1. The Cross-Modal Modality Gap
    Panchromatic (single high-resolution band) and Multispectral (multiple lower-resolution spectral bands) images are captured by different sensors. Their raw pixel representations are fundamentally unaligned. Passing them through standard models results in completely different feature vectors, preventing direct retrieval.
    
    #### 2. Supervised Contrastive Learning (SupCon)
    Instead of pair-only contrastive learning (which only aligns `PAN[i]` with `MUL[i]`), **Supervised Contrastive Learning** leverages the class labels. It forces all PAN and MUL embeddings that share the same class label (e.g., all Forests) to map close to each other in the shared 128D space, while pushing different categories apart.
    
    #### 3. FAISS Indexing (Inner Product)
    **FAISS (Facebook AI Similarity Search)** is utilized for fast similarity search. By normalizing the 128D embeddings to unit length and using dynamic index choices (FlatIP/HNSW/IVF PQ), similarity searches perform **Cosine Similarity calculations** on CPU in less than a millisecond, making it perfect for real-time remote sensing operations.
    """)

# ----------------- SESSION CLEANUP & MEMORY MANAGEMENT -----------------
# Temporary image arrays are not stored in persistent session state or caches.
# They are automatically reclaimed by Python's garbage collector naturally after each rerun.
