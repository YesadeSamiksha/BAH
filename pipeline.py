import os
import sys
import argparse
import subprocess

# Ensure UTF-8 output encoding for terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add current directory to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATASET_PATH,
    BACKBONE,
    FAISS_INDEX,
    get_experiment_dir,
    STAGE_DEPENDENCIES,
    verify_cache,
    update_pipeline_manifest,
    FAISS_DIR,
    MODEL_DIR
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define scripts associated with each stage
STAGE_SCRIPTS = {
    "baseline_embeddings": ["extract_embeddings.py"],
    "training_complete": ["train_contrastive.py"],
    "contrastive_embeddings": ["evaluate_contrastive.py"],
    "faiss_ready": ["build_faiss.py", "evaluate_contrastive.py"],
    "metrics_ready": ["precompute_metrics.py"],
    "retrieval_complete": ["retrieve.py"],
    "visualizations_complete": ["visualize_embeddings.py"]
}

def get_stage_status_and_reason(stage_name):
    # Special case: dataset_ready
    if stage_name == "dataset_ready":
        if os.path.exists(DATASET_PATH):
            return "✔ Cached", "Dataset file verified"
        return "⚠ Cache Invalid", f"Dataset file missing at {DATASET_PATH}"
        
    dependencies = STAGE_DEPENDENCIES.get(stage_name, [])
    unmet_deps = []
    for dep in dependencies:
        dep_cached, _ = verify_cache(dep)
        if not dep_cached:
            unmet_deps.append(dep)
            
    if unmet_deps:
        return "Pending", f"Requires: {', '.join(unmet_deps)}"
        
    is_cached, reason = verify_cache(stage_name)
    if is_cached:
        if reason and "Manifest synchronization issue" in reason:
            return "⚠ Sync Issue", reason
        return "✔ Cached", "All output files verified"
    else:
        return "⚠ Cache Invalid", reason or "Output files missing or configuration mismatch"

def print_status():
    import config
    config.READ_ONLY_MODE = True
    print("=" * 85)
    print("                       CROSS-MODAL PIPELINE STATUS REPORT")
    print("=" * 85)
    print(f"Backbone Architecture : {BACKBONE}")
    print(f"FAISS Indexing Type   : {FAISS_INDEX}")
    print(f"Dataset Path          : {DATASET_PATH}")
    print(f"Experiment Directory  : {get_experiment_dir()}")
    onnx_file = os.path.join(MODEL_DIR, "best_model.onnx")
    onnx_status = "Available" if os.path.exists(onnx_file) else "Not Generated"
    print(f"ONNX Export           : {onnx_status}")
    print("-" * 85)
    print(f"{'Stage Name':<25} | {'Status':<15} | {'Details / Diagnostic Reasons'}")
    print("-" * 85)
    
    stages = ["dataset_ready", "baseline_embeddings", "training_complete", "contrastive_embeddings", 
              "faiss_ready", "metrics_ready", "retrieval_complete", "visualizations_complete"]
              
    for stage in stages:
        status, reason = get_stage_status_and_reason(stage)
        print(f"{stage:<25} | {status:<15} | {reason}")
    print("=" * 85)

def run_stage_with_dependencies(stage_name, visited=None):
    if visited is None:
        visited = set()
    if stage_name in visited:
        return
    visited.add(stage_name)
    
    # 1. Resolve dependencies first
    dependencies = STAGE_DEPENDENCIES.get(stage_name, [])
    for dep in dependencies:
        is_cached, _ = verify_cache(dep)
        if not is_cached:
            print(f"🔄 Stage '{stage_name}' depends on '{dep}' which is not cached.")
            print(f"   Running prerequisite stage '{dep}' first...")
            run_stage_with_dependencies(dep, visited)
            
    # 2. Run the stage if not cached
    is_cached, reason = verify_cache(stage_name)
    if is_cached:
        if reason and "Manifest synchronization issue" in reason:
            print(f"ℹ Synchronizing manifest for stage '{stage_name}' (outputs exist and are valid).")
            update_pipeline_manifest(stage_name, True)
        print(f"✔ Stage '{stage_name}' is already cached. Skipping.")
        return
        
    print(f"\n🚀 Running stage: '{stage_name}'...")
    if stage_name == "dataset_ready":
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset missing at {DATASET_PATH}. Please check configuration.")
        return
        
    scripts = STAGE_SCRIPTS.get(stage_name, [])
    for script in scripts:
        # Optimization for faiss_ready
        if script == "build_faiss.py":
            pan_idx = os.path.join(FAISS_DIR, "pan_index.bin")
            mul_idx = os.path.join(FAISS_DIR, "mul_index.bin")
            if os.path.exists(pan_idx) and os.path.exists(mul_idx):
                # Baseline indices are already built, we don't need to rebuild them.
                # Contrastive indices will be built by evaluate_contrastive.py.
                continue
                
        cmd = [sys.executable, "-X", "utf8", script]
        print(f"Executing: {' '.join(cmd)}")
        
        res = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if res.returncode != 0:
            print(f"❌ Stage '{stage_name}' failed at script '{script}' (exit code: {res.returncode})")
            sys.exit(res.returncode)
            
    # Verify cache after run
    is_cached, reason = verify_cache(stage_name)
    if not is_cached:
        print(f"❌ Error: Stage '{stage_name}' execution finished but cache verification failed! Reason: {reason}")
        sys.exit(1)
    else:
        # If it was verified but was a sync issue, heal it now.
        if reason and "Manifest synchronization issue" in reason:
            print(f"ℹ Synchronizing manifest for stage '{stage_name}' (outputs exist and are valid).")
            update_pipeline_manifest(stage_name, True)
        print(f"✔ Stage '{stage_name}' verified successfully.\n")

def main():
    parser = argparse.ArgumentParser(description="Cross-Modal Satellite Image Retrieval Pipeline Runner")
    parser.add_argument("--extract", action="store_true", help="Extract baseline embeddings")
    parser.add_argument("--train", action="store_true", help="Train Supervised Contrastive Model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate contrastive model & generate metrics/plots")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline from scratch")
    parser.add_argument("--status", action="store_true", help="Display current pipeline status and cache report")
    
    args = parser.parse_args()
    
    # If no arguments are passed, show help
    if not (args.extract or args.train or args.evaluate or args.all or args.status):
        parser.print_help()
        print("\n--- Pipeline Status ---")
        print_status()
        return
        
    if args.status:
        print_status()
        return
        
    if args.all:
        print("Running entire pipeline...")
        stages = ["baseline_embeddings", "faiss_ready", "training_complete", "contrastive_embeddings", 
                  "metrics_ready", "retrieval_complete", "visualizations_complete"]
        for stage in stages:
            run_stage_with_dependencies(stage)
        print("🎉 Entire pipeline executed and verified successfully!")
        print_status()
        return
        
    if args.extract:
        run_stage_with_dependencies("baseline_embeddings")
        
    if args.train:
        run_stage_with_dependencies("training_complete")
        
    if args.evaluate:
        # Run evaluation stages
        eval_stages = ["contrastive_embeddings", "faiss_ready", "metrics_ready", "retrieval_complete", "visualizations_complete"]
        for stage in eval_stages:
            run_stage_with_dependencies(stage)
            
    print_status()

if __name__ == "__main__":
    main()
