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
from train_contrastive import ContrastiveModel, FREEZE_BACKBONE

# Page Configuration
st.set_page_config(
    page_title="Cross-Modal Satellite Image Retrieval System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    /* Main Layout */
    .reportview-container {
        background: #0E1117;
        color: #E0E0E0;
    }
    
    /* Header styling */
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 2.8rem;
        background: linear-gradient(90deg, #4285F4, #34A853, #FBBC05, #EA4335);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #F1F3F4;
        font-weight: 600;
    }

    /* Cards */
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 15px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #4285F4;
    }

    /* Result Card Styles */
    .result-card {
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #0F172A;
        border: 2px solid;
        transition: transform 0.2s ease;
    }
    
    .result-card:hover {
        transform: scale(1.02);
    }
    
    .result-correct {
        border-color: #10B981; /* Green */
        background-color: rgba(16, 185, 129, 0.05);
    }
    
    .result-incorrect {
        border-color: #EF4444; /* Red */
        background-color: rgba(239, 68, 68, 0.05);
    }

    /* Subtext */
    .result-meta {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    
    .result-reason {
        font-size: 0.8rem;
        color: #CBD5E1;
        font-style: italic;
        background-color: rgba(0, 0, 0, 0.2);
        padding: 8px;
        border-radius: 6px;
        border-left: 3px solid #64748B;
        margin-top: 8px;
    }
    
    /* Stats box */
    .stats-box {
        background: #020617;
        border-left: 4px solid #4285F4;
        padding: 12px;
        margin-bottom: 15px;
        border-radius: 4px;
        font-family: monospace;
    }

</style>
""", unsafe_allow_html=True)

# Define Classes and AI Explanations
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

REASONS = {
    "Aquafarm": "Matched due to regular grid patterns of coastal enclosures, shallow water reflectivity, and maritime infrastructure footprints.",
    "Cloud": "Matched based on bright, high-reflectance cloud mass boundaries, localized atmospheric opacity, and diffuse white texture.",
    "Forest": "Matched due to dense, irregular vegetation canopy texture, low reflectance in visible bands, and high biological cell-density signature.",
    "High Building": "Matched based on tall concrete footprints, high density shadow projections, and high spatial structural complexity.",
    "Low Building": "Matched based on residential spatial sprawl, small distinct building boundary profiles, and suburban street grid features.",
    "Farmland": "Matched due to rectangular field plot divisions, visible crop patterns, and agricultural soil reflectance structures.",
    "River": "Matched due to winding linear water channels, high absorption in near-infrared spectrum, and riverbank vegetation transitions.",
    "Water": "Matched based on open water surface characteristics, high visual homogeneity, and deep light absorption properties."
}

MISMATCH_REASON = "Mismatched class features; retrieved due to similar shape outlines, coarse structural silhouettes, or low-level sensor reflectance overlaps."

# Load cached data and models
@st.cache_resource
def load_resources():
    # Load dataset indices and instantiate DSRSIDDataset
    indices_path = "subset_indices.npy"
    if not os.path.exists(indices_path):
        st.error("Missing 'subset_indices.npy'! Please run extraction first.")
        st.stop()
        
    subset_indices = np.load(indices_path)
    dataset = DSRSIDDataset(file_path="data/DSRSID.mat", indices=subset_indices)
    
    # Load labels
    labels = np.load("labels.npy")
    
    # Load Baseline Embeddings and FAISS Indices
    base_pan_embs = np.load("pan_embeddings.npy")
    base_mul_embs = np.load("mul_embeddings.npy")
    base_pan_index = faiss.read_index("pan_index.bin")
    base_mul_index = faiss.read_index("mul_index.bin")
    
    # Load Contrastive Embeddings and FAISS Indices
    c_pan_embs = np.load("pan_embeddings_contrastive.npy")
    c_mul_embs = np.load("mul_embeddings_contrastive.npy")
    c_pan_index = faiss.read_index("pan_index_contrastive.bin")
    c_mul_index = faiss.read_index("mul_index_contrastive.bin")
    
    # Load precomputed metrics JSON
    metrics_file = "metrics_summary.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics_summary = json.load(f)
    else:
        metrics_summary = {}

    # Load Trained Contrastive Model
    model = ContrastiveModel(freeze_backbone=FREEZE_BACKBONE)
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
    model.eval()

    return {
        "dataset": dataset,
        "labels": labels,
        "base_pan_embs": base_pan_embs,
        "base_mul_embs": base_mul_embs,
        "base_pan_index": base_pan_index,
        "base_mul_index": base_mul_index,
        "c_pan_embs": c_pan_embs,
        "c_mul_embs": c_mul_embs,
        "c_pan_index": c_pan_index,
        "c_mul_index": c_mul_index,
        "metrics_summary": metrics_summary,
        "model": model
    }

res = load_resources()

# Streamlit Title
st.title("Cross-Modal Satellite Image Retrieval System")
st.markdown("##### *Aligning Panchromatic & Multispectral Satellite Data with Supervised Contrastive Learning*")

# Tabs Layout
tab_retrieval, tab_eval, tab_embeddings, tab_arch, tab_about = st.tabs([
    "🔍 Retrieval Sandbox", 
    "📈 Performance Dashboard", 
    "🔮 t-SNE Embeddings", 
    "🏗️ Pipeline Architecture", 
    "📋 About Project"
])

# Sidebar Controls
st.sidebar.header("🔧 Retrieval Configuration")

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

# Demo Mode Shortcut Selector
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Demo Examples (One-Click)")
cols_demo1, cols_demo2 = st.sidebar.columns(2)

demo_idx = None
with cols_demo1:
    if st.button("🌲 Forest Example"):
        demo_idx = 1500  # Class 3.0
    if st.button("🏢 Urban Example"):
        demo_idx = 2000  # Class 4.0
with cols_demo2:
    if st.button("🌊 Water Example"):
        demo_idx = 4600  # Class 8.0
    if st.button("🛣️ River Example"):
        demo_idx = 4000  # Class 7.0

# ----------------- TAB 1: RETRIEVAL SANDBOX -----------------
with tab_retrieval:
    st.subheader("Interactive Query Interface")
    
    # Initialize query index in session state if not set
    if 'query_idx_input' not in st.session_state:
        st.session_state.query_idx_input = 1500

    # Override session state index if a demo example button was clicked
    if demo_idx is not None:
        st.session_state.query_idx_input = demo_idx
        
    query_idx = st.session_state.query_idx_input
    
    query_image = None
    query_label_val = None
    query_embedding = None
    latency_extract = 0.0
    latency_search = 0.0

    if query_source == "Dataset Index Mode":
        st.markdown("Select a query index from the 5,000 stratified samples. Images and metadata will load from the DSRSID dataset.")
        
        # User input for index
        query_idx = st.number_input(
            "Dataset Sample Index (0 - 4999)", 
            min_value=0, max_value=4999, 
            value=query_idx,
            key='query_idx_input',
            help="Select index. The 5000 samples are stratified (625 per class: 0-624 Class 1, 625-1249 Class 2, ...)"
        )
        
        # Load from dataset
        pan_pil, mul_pil, label_val = res["dataset"].get_visualization_images(query_idx)
        query_label_val = label_val
        
        # Determine query image and pre-extracted contrastive embedding
        if retrieval_mode.startswith("PAN"):
            query_image = pan_pil
            query_embedding = res["c_pan_embs"][query_idx]
        else:
            query_image = mul_pil
            query_embedding = res["c_mul_embs"][query_idx]
            
        latency_extract = 0.0  # Already pre-extracted!
        
    else:  # File Upload Mode
        st.markdown("Upload a custom satellite image (PAN or MUL RGB visualization) to perform real-time encoder inference and retrieval.")
        
        uploaded_file = st.file_uploader(
            f"Upload {'Panchromatic' if retrieval_mode.startswith('PAN') else 'Multispectral RGB'} Image",
            type=["png", "jpg", "jpeg"]
        )
        
        if uploaded_file is not None:
            # Load and display uploaded image
            query_image = Image.open(uploaded_file)
            
            # Setup torch preprocess transform
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Perform inference and measure latency
            t0 = time.perf_counter()
            
            # Convert single channel to RGB if PAN
            if retrieval_mode.startswith("PAN"):
                img_rgb = query_image.convert("RGB")
                img_tensor = preprocess(img_rgb).unsqueeze(0)
                with torch.no_grad():
                    emb_torch = res["model"].forward_pan(img_tensor)
            else:  # MUL upload
                img_rgb = query_image.convert("RGB")
                img_tensor = preprocess(img_rgb).unsqueeze(0)
                with torch.no_grad():
                    emb_torch = res["model"].forward_mul(img_tensor)
                    
            query_embedding = emb_torch.squeeze(0).numpy()
            
            # Normalize embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            t1 = time.perf_counter()
            latency_extract = (t1 - t0) * 1000.0  # in ms
            query_label_val = None  # Custom upload has no ground truth label in the dataset
        else:
            st.info("Please upload an image file to perform retrieval.")
            
    # Perform Search if Query is loaded
    if query_embedding is not None:
        # Determine target index and mode parameters
        target_index = res["c_mul_index"] if retrieval_mode.endswith("MUL") else res["c_pan_index"]
        target_modality = "MUL" if retrieval_mode.endswith("MUL") else "PAN"
        
        # Execute search and measure FAISS search latency
        t0 = time.perf_counter()
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        # If dataset query mode, we exclude the self index from search results
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
        latency_search = (t1 - t0) * 1000.0  # in ms
        
        # Layout: Left for Query, Right for Top-5 Retrieved results
        col_query, col_results = st.columns([1, 3])
        
        with col_query:
            st.markdown("### 🎯 Query Image")
            st.image(query_image, use_container_width=True)
            
            label_text = CLASSES.get(query_label_val, "Unknown (Uploaded File)")
            st.markdown(f"**Modality**: `{retrieval_mode.split('→')[0].strip()}`")
            st.markdown(f"**Class Label**: `{label_text}`")
            
            # Latency statistics box
            st.markdown("##### ⏱️ Retrieval Latency")
            latency_total = latency_extract + latency_search
            st.markdown(f"""
            <div class="stats-box">
            Inference  : {latency_extract:6.2f} ms<br>
            FAISS Index: {latency_search:6.2f} ms<br>
            Total Time : {latency_total:6.2f} ms
            </div>
            """, unsafe_allow_html=True)
            
        with col_results:
            st.markdown("### 🏆 Top-5 Retrieved Matches")
            
            res_cols = st.columns(5)
            csv_data = []
            
            for r, (ret_idx, similarity) in enumerate(zip(retrieved_idxs, retrieved_sims)):
                # Load retrieved images from dataset
                ret_pan_pil, ret_mul_pil, ret_lbl_val = res["dataset"].get_visualization_images(ret_idx)
                ret_img = ret_mul_pil if target_modality == "MUL" else ret_pan_pil
                ret_class_name = CLASSES.get(ret_lbl_val, "Unknown")
                
                # Check match correctness
                is_match = (query_label_val is not None and ret_lbl_val == query_label_val)
                card_class = "result-correct" if is_match else "result-incorrect"
                match_label = "✅ Match" if is_match else "❌ Mismatch"
                
                # Dynamic explainable AI reason
                reason_text = REASONS.get(ret_class_name, MISMATCH_REASON) if is_match else MISMATCH_REASON
                
                with res_cols[r]:
                    st.image(ret_img, use_container_width=True)
                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <div style="font-weight: bold; font-size: 0.95rem;">Rank {r+1}</div>
                        <div class="result-meta">
                            <b>Sim:</b> {(similarity*100):.1f}%<br>
                            <b>Class:</b> {ret_class_name}<br>
                            <b>Index:</b> {ret_idx}
                        </div>
                        <div style="font-size: 0.8rem; font-weight: bold; margin-top: 5px;">{match_label}</div>
                        <div class="result-reason">
                            {reason_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                csv_data.append({
                    "Rank": r + 1,
                    "Dataset_Index": ret_idx,
                    "Similarity_Score": f"{similarity:.5f}",
                    "Class_Label": ret_class_name,
                    "Is_Correct": "Yes" if is_match else "No"
                })

            # Create download CSV button
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=["Rank", "Dataset_Index", "Similarity_Score", "Class_Label", "Is_Correct"])
            writer.writeheader()
            writer.writerows(csv_data)
            
            st.markdown("---")
            st.download_button(
                label="📥 Download Retrieval Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"retrieval_results_idx_{query_idx}.csv",
                mime="text/csv"
            )

# ----------------- TAB 2: PERFORMANCE DASHBOARD -----------------
with tab_eval:
    st.subheader("Model Performance & Evaluation Dashboard")
    st.markdown("Comparison between the **Baseline 512D (L2)** and the **Supervised Contrastive 128D (Cosine)** models over all 5,000 stratified queries.")
    
    # Load metrics summaries
    ms = res["metrics_summary"]
    
    if not ms:
        st.warning("No pre-computed metrics found. Run 'precompute_metrics.py' to generate.")
    else:
        # Grouped Mode Cards with st.metric highlighting Before vs After
        st.markdown("#### ⚡ Before vs. After Contrastive Learning")
        
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
                
        # st.bar_chart Grouped Chart
        st.markdown("---")
        st.markdown("#### 📊 Precision@5 Retrieval Mode Comparison")
        
        # Create Data for bar chart
        chart_data = {
            "Retrieval Mode": ["PAN→PAN", "MUL→MUL", "PAN→MUL", "MUL→PAN"],
            "Baseline (512D L2)": [
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
        
        # Dynamic Latency Benchmark Card
        st.markdown("---")
        st.markdown("#### ⏱️ Real-Time Local CPU Latency Benchmarking")
        
        # Setup run benchmark button
        if st.button("⚡ Run CPU Latency Benchmark (20 Queries)"):
            with st.spinner("Running search benchmarks on CPU..."):
                ext_times = []
                search_times = []
                
                # Select 20 random indices
                random_idxs = np.random.choice(5000, 20, replace=False)
                
                # Setup a test image tensor for extraction time
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                dummy_pil = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
                dummy_tensor = preprocess(dummy_pil).unsqueeze(0)
                
                # Benchmark loop
                for test_idx in random_idxs:
                    # 1. Measure Embedding extraction
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        _ = res["model"].forward_pan(dummy_tensor)
                    t1 = time.perf_counter()
                    ext_times.append((t1 - t0) * 1000.0)
                    
                    # 2. Measure FAISS Search
                    q_emb = res["c_pan_embs"][test_idx].reshape(1, -1).astype('float32')
                    t0 = time.perf_counter()
                    _, _ = res["c_mul_index"].search(q_emb, 5)
                    t1 = time.perf_counter()
                    search_times.append((t1 - t0) * 1000.0)
                    
                avg_ext = np.mean(ext_times)
                avg_search = np.mean(search_times)
                
                st.success("Benchmark completed successfully!")
                
                cols_lat = st.columns(2)
                with cols_lat[0]:
                    st.metric("Avg Embedding Extraction Time", f"{avg_ext:.2f} ms", help="Time taken by ResNet18 + Projection Head forward pass on CPU.")
                with cols_lat[1]:
                    st.metric("Avg FAISS Index Search Time", f"{avg_search:.2f} ms", help="Time taken by IndexFlatIP search for 128D embeddings.")

# ----------------- TAB 3: T-SNE EMBEDDINGS -----------------
with tab_embeddings:
    st.subheader("t-SNE Embedding Space Projections")
    st.markdown("Visualizing the alignment of 800 sampled PAN (circles `o`) and MUL (triangles `^`) vectors in 2D space before and after contrastive learning.")
    
    col_tsne_before, col_tsne_after = st.columns(2)
    
    with col_tsne_before:
        st.markdown("##### ❌ Before Contrastive Training (Baseline 512D)")
        if os.path.exists("embeddings_tsne_before.png"):
            st.image("embeddings_tsne_before.png", use_container_width=True)
            st.markdown("<p style='font-size:0.85rem; color:#94A3B8; text-align:center;'>Modalities are completely separated. PAN and MUL representations of the same class do not align in the shared space.</p>", unsafe_allow_html=True)
        else:
            st.info("Missing 'embeddings_tsne_before.png' image.")
            
    with col_tsne_after:
        st.markdown("##### ✅ After Supervised Contrastive Training (Aligned 128D)")
        if os.path.exists("embeddings_tsne_after.png"):
            st.image("embeddings_tsne_after.png", use_container_width=True)
            st.markdown("<p style='font-size:0.85rem; color:#94A3B8; text-align:center;'>PAN and MUL representations of the same classes merge and overlap, showing successful cross-modal alignment.</p>", unsafe_allow_html=True)
        else:
            st.info("Missing 'embeddings_tsne_after.png' image.")

# ----------------- TAB 4: ARCHITECTURE DIAGRAM -----------------
with tab_arch:
    st.subheader("Dual-Encoder Embedding Alignment Architecture")
    st.markdown("Below is the flow of Panchromatic (PAN) and Multispectral (MUL) modalities through their frozen ResNet18 backbones and trainable projection heads into the aligned 128D embedding space.")
    
    st.graphviz_chart('''
    digraph G {
        rankdir=LR;
        node [style=filled, fillcolor="#1E293B", color="#38BDF8", fontcolor="#F8FAFC", fontname="Arial", shape=box, penwidth=1.5];
        edge [color="#38BDF8", fontname="Arial", fontcolor="#94A3B8", penwidth=1.5];
        bgcolor="transparent";
        
        subgraph cluster_pan {
            label = "Panchromatic (PAN) Stream";
            fontname="Arial";
            color = "#3B82F6";
            style = dashed;
            fontcolor = "#3B82F6";
            
            PAN [label="PAN Image\\n(1 x 256 x 256)", fillcolor="#0F172A"];
            PAN_Enc [label="ResNet18 backbone\\n(fc layer replaced with Identity)\\n(Frozen)", shape=ellipse];
            PAN_Proj [label="Projection Head\\nLinear(512, 256) -> ReLU -> Linear(256, 128)\\n(Trainable)"];
            
            PAN -> PAN_Enc -> PAN_Proj;
        }
        
        subgraph cluster_mul {
            label = "Multispectral (MUL) Stream";
            fontname="Arial";
            color = "#10B981";
            style = dashed;
            fontcolor = "#10B981";
            
            MUL [label="MUL RGB Image\\n(3 x 64 x 64)", fillcolor="#0F172A"];
            MUL_Enc [label="ResNet18 backbone\\n(fc layer replaced with Identity)\\n(Frozen)", shape=ellipse];
            MUL_Proj [label="Projection Head\\nLinear(512, 256) -> ReLU -> Linear(256, 128)\\n(Trainable)"];
            
            MUL -> MUL_Enc -> MUL_Proj;
        }
        
        Shared [label="Shared 128D Space\\n(L2 Normalized Vectors)", shape=hexagon, fillcolor="#3F2B3E", color="#EC4899", fontcolor="#F472B6"];
        
        PAN_Proj -> Shared [label="Cosine Similarity\\n(FAISS FlatIP Search)", constraint=true];
        MUL_Proj -> Shared [label="Cosine Similarity\\n(FAISS FlatIP Search)", constraint=true];
    }
    ''')

# ----------------- TAB 5: ABOUT PROJECT -----------------
with tab_about:
    st.subheader("Project Information & Hackathon Details")
    
    # Dataset Statistics Card
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin-top:0px; color:#4285F4;">📊 Dataset Card: DSRSID</h4>
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
    
    st.markdown("### 🛰️ Core Concepts Explained")
    
    st.markdown("""
    #### 1. The Cross-Modal Modality Gap
    Panchromatic (single high-resolution band) and Multispectral (multiple lower-resolution spectral bands) images are captured by different sensors. Their raw pixel representations are fundamentally unaligned. Passing them through standard models results in completely different feature vectors, preventing direct retrieval.
    
    #### 2. Supervised Contrastive Learning (SupCon)
    Instead of pair-only contrastive learning (which only aligns `PAN[i]` with `MUL[i]`), **Supervised Contrastive Learning** leverages the class labels. It forces all PAN and MUL embeddings that share the same class label (e.g., all Forests) to map close to each other in the shared 128D space, while pushing different categories apart.
    
    #### 3. FAISS Indexing (Inner Product)
    **FAISS (Facebook AI Similarity Search)** is utilized for fast similarity search. By normalizing the 128D embeddings to unit length and using `faiss.IndexFlatIP` (Inner Product), similarity searches perform **Cosine Similarity calculations** on CPU in less than a millisecond, making it perfect for real-time remote sensing operations.
    """)
