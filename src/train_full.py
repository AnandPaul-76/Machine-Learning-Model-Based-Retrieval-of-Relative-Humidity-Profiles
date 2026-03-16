import os
import re
import argparse
import numpy as np
import joblib
import lightgbm as lgb
import h5py
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
import gc
import time


# ===================== CONFIGURATION =====================

# Dataset base directory
# Priority:
# 1. Environment variable MEGATROPIQUES_DATA
# 2. Default local folder "data"
BASE_DIR = os.getenv("MEGATROPIQUES_DATA", "data")

# Cache and model directories
NPZ_DIR = "npz_cache"
MODEL_DIR = "models"

# BATCH CONTROL
BATCH_SIZE = 20
MAX_ORBITS = None
STOP_AT_BATCH = 69

# Ensure folders exist
os.makedirs(NPZ_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

# Regex to extract orbit ID
ORBIT_RE = re.compile(r"_([0-9]{5})_")

# =====================  UTILITIES =====================
def get_orbit_id(name):
    m = ORBIT_RE.search(name)
    return m.group(1) if m else None

def scan_files():
    l1a, l2a = {}, {}
    for root, _, files in os.walk(BASE_DIR):
        for f in files:
            if not f.endswith(".h5"): continue
            orbit = get_orbit_id(f)
            if not orbit: continue
            path = os.path.join(root, f)
            if "L1A" in f: l1a[orbit] = path
            elif "L2A" in f: l2a[orbit] = path
    return l1a, l2a

# =====================  PREPROCESS (UNCHANGED) =====================
def preprocess_orbit(orbit, l1a_file, l2a_file):
    out_npz = os.path.join(NPZ_DIR, f"{orbit}.npz")
    if os.path.exists(out_npz): return

    try:
        with h5py.File(l1a_file, "r") as f:
            sd = f["ScienceData"]
            TB = np.stack([
                sd["TB_Samples_S2"][:], sd["TB_Samples_S3"][:],
                sd["TB_Samples_S4"][:], sd["TB_Samples_S5"][:]
            ], axis=-1) * 0.01
            lat1 = sd["Latitude_Samples"][:] * 0.01
            lon1 = sd["Longitude_Samples"][:] * 0.01
            inc  = sd["IncidenceAngle_Samples"][:] * 0.01
            TBm = uniform_filter(TB, size=(3, 3, 1))
            mask_l1 = np.all(np.isfinite(TB), axis=-1) & (lat1 > -90)

        X_flat = np.column_stack([TB[mask_l1], TBm[mask_l1], lat1[mask_l1], lon1[mask_l1], inc[mask_l1]]).astype(np.float32)
        coords1 = np.column_stack([lat1[mask_l1], lon1[mask_l1]])

        with h5py.File(l2a_file, "r") as f:
            sd = f["ScienceData"]
            lat2 = sd["Latitude"][:] * 0.01
            lon2 = sd["Longitude"][:] * 0.01
            RH_raw = {
                "L2": sd["RelativeHumidity_L2"][:] * 0.01,
                "L3": sd["RelativeHumidity_L3"][:] * 0.01,
                "L4": sd["RelativeHumidity_L4"][:] * 0.01,
                "L5": sd["RelativeHumidity_L5"][:] * 0.01
            }

        mask_l2_geo = (lat2 > -90) & (lat2 < 90)
        coords2 = np.column_stack([lat2[mask_l2_geo], lon2[mask_l2_geo]])
        for k in RH_raw: RH_raw[k] = RH_raw[k][mask_l2_geo]

        tree = cKDTree(coords2)
        dist, idx = tree.query(coords1, k=1)
        valid_match = dist < 0.05
        
        X_final = X_flat[valid_match]
        matched_indices = idx[valid_match]
        save_dict = {"X": X_final}
        
        for layer_name, rh_data in RH_raw.items():
            y_values = rh_data[matched_indices]
            y_clean = np.where((y_values >= 0) & (y_values <= 100), y_values, np.nan)
            save_dict[f"Y_{layer_name}"] = y_clean.astype(np.float32)

        np.savez_compressed(out_npz, **save_dict)
        print(f"✅ Processed {orbit}")
    except Exception as e:
        print(f"❌ Failed {orbit}: {e}")

# =====================  RESUMABLE TRAIN =====================
def train_resumable(rh_layer):
    # Checkpoint File Paths
    ckpt_model_path = os.path.join(MODEL_DIR, f"checkpoint_{rh_layer}.txt")
    ckpt_meta_path = os.path.join(MODEL_DIR, f"checkpoint_{rh_layer}_meta.txt")
    final_model_path = os.path.join(MODEL_DIR, f"RH_{rh_layer}.pkl")

    print(f"\n[RESUMABLE TRAINING] Target: {rh_layer}")
    
    files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])
    if MAX_ORBITS: files = files[:MAX_ORBITS]
    
    total_files = len(files)
    num_batches = int(np.ceil(total_files / BATCH_SIZE))

    # --- 1. CHECK FOR RESUME ---
    start_batch_idx = 0
    model = None

    if os.path.exists(ckpt_model_path) and os.path.exists(ckpt_meta_path):
        try:
            with open(ckpt_meta_path, "r") as f:
                saved_batch = int(f.read().strip())
            
            # --- AUTO-STOP PRE-CHECK ---
            if saved_batch >= STOP_AT_BATCH:
                print(f"✅ Target of {STOP_AT_BATCH} batches already reached for {rh_layer}!")
                bst = lgb.Booster(model_file=ckpt_model_path)
                joblib.dump(bst, final_model_path)
                print(f"✅ Final Model Saved: {final_model_path}")
                return
            
            print(f"🔄 FOUND CHECKPOINT! Resuming from Batch {saved_batch + 1}...")
            start_batch_idx = saved_batch
            # We don't load the model into RAM yet; we pass the filename to init_model
        except Exception as e:
            print(f"⚠️ Checkpoint corrupted ({e}). Starting fresh.")

    # --- 2. TRAIN LOOP ---
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 100,
        "max_depth": 15,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "num_threads": 6,
        "verbosity": 1  # Show logs!
    }

    print(f" Training {num_batches} batches, but stopping at Batch {STOP_AT_BATCH}...")

    for i in range(0, total_files, BATCH_SIZE):
        current_batch_num = (i // BATCH_SIZE) + 1
        
        # SKIP LOGIC: If we already finished this batch, skip it.
        if current_batch_num <= start_batch_idx:
            continue

        # --- AUTO-STOP TRIGGER ---
        if current_batch_num > STOP_AT_BATCH:
            print(f"\n🎯 Reached target of {STOP_AT_BATCH} batches! Stopping early.")
            break

        batch_files = files[i : i + BATCH_SIZE]
        print(f"\n Batch {current_batch_num}/{num_batches} | Loading {len(batch_files)} orbits...")
        
        X_batch = []
        y_batch = []

        # Load Data
        for f in batch_files:
            try:
                data = np.load(os.path.join(NPZ_DIR, f))
                X = data["X"]
                y = data[f"Y_{rh_layer}"]
                valid = np.isfinite(y)
                if valid.sum() > 0:
                    X_batch.append(X[valid])
                    y_batch.append(y[valid])
            except Exception as e:
                print(f"   ⚠️ Error {f}: {e}")

        if not X_batch: continue

        X_batch = np.vstack(X_batch)
        y_batch = np.hstack(y_batch)
        
        print(f"   Training on {X_batch.shape[0]:,} pixels...")

        # Create Dataset
        dtrain = lgb.Dataset(X_batch, label=y_batch, free_raw_data=True)

        # RESUME LOGIC: Pass previous model file if it exists
        init_model_arg = ckpt_model_path if os.path.exists(ckpt_model_path) else None
        
        # Train
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=50,
            init_model=init_model_arg,  # <--- MAGIC LINE
            keep_training_booster=True,
            callbacks=[lgb.log_evaluation(100)]
        )

        # SAVE CHECKPOINT (Crucial Step)
        model.save_model(ckpt_model_path)
        with open(ckpt_meta_path, "w") as f:
            f.write(str(current_batch_num))
        
        print(f" Checkpoint Saved (Batch {current_batch_num} Done)")

        # Cleanup RAM
        del X_batch, y_batch, dtrain
        gc.collect()

    # --- 3. FINALIZE ---
    print(f"\n Training Complete!")
    # Load final checkpoint and save as PKL for compatibility
    bst = lgb.Booster(model_file=ckpt_model_path)
    joblib.dump(bst, final_model_path)
    print(f"✅ Final Model Saved: {final_model_path}")

# =====================  RUNNER =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preprocess", "train"], required=True)
    parser.add_argument("--rh", choices=["L2", "L3", "L4", "L5"])
    args = parser.parse_args()

    if args.mode == "preprocess":
        l1a, l2a = scan_files()
        common = sorted(set(l1a) & set(l2a))
        print(f"Found {len(common)} pairs.")
        for orbit in common: preprocess_orbit(orbit, l1a[orbit], l2a[orbit])
            
    elif args.mode == "train":
        if not args.rh:
            print("❌ Error: --rh required")
        else:
            train_resumable(args.rh)