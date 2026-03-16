import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import time

# ================= CONFIG =================
RH_LAYER = "L2"
MODEL_PATH = f"models/checkpoint_{RH_LAYER}.txt"
NPZ_DIR = "npz_cache"
FILES_TO_TEST = 5  # <-- Changed to 5 for lightning-fast RAM-friendly testing!
# ==========================================

def validate_brain_fast():
    print(f" Loading {RH_LAYER} Model Brain...")
    if not os.path.exists(MODEL_PATH):
        print(f" Cannot find model at {MODEL_PATH}")
        return
        
    model = lgb.Booster(model_file=MODEL_PATH)

    all_files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')])
    test_files = all_files[-FILES_TO_TEST:]
    
    print(f" Testing on {len(test_files)} unseen files...")

    y_true_all = []
    y_pred_all = []

    # File-by-file progress so you know it's not stuck!
    for i, f in enumerate(test_files):
        print(f"   Processing file {i+1}/{FILES_TO_TEST}: {f}...", end="")
        start_time = time.time()
        try:
            data = np.load(os.path.join(NPZ_DIR, f))
            X = data['X']
            y = data[f'Y_{RH_LAYER}']
            
            valid = np.isfinite(y)
            X_clean = X[valid]
            y_clean = y[valid]
            
            if len(y_clean) > 0:
                preds = model.predict(X_clean)
                y_true_all.append(y_clean)
                y_pred_all.append(preds)
                print(f" Done! ({len(y_clean):,} pixels in {time.time()-start_time:.1f}s)")
            else:
                print(" Skipped (No valid pixels).")
        except Exception as e:
            print(f" Failed: {e}")

    if not y_true_all:
        print(" No data was processed.")
        return

    print("\n Crunching final math...")
    # RAM-friendly joining
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    total_pixels = len(y_true_all)
    
    # Calculate Metrics
    mae = mean_absolute_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    r2 = r2_score(y_true_all, y_pred_all)
    bias = np.mean(y_pred_all - y_true_all)
    pearson_corr, _ = pearsonr(y_true_all, y_pred_all)

    print("\n" + "="*50)
    print(" READY ")
    print("="*50)
    print(f"Total Pixels Tested : {total_pixels:,}")
    print(f"Mean Abs Error (MAE): {mae:.3f} %")
    print(f"Root Mean Sq (RMSE) : {rmse:.3f} %")
    print(f"Overall Bias        : {bias:.3f} %")
    print(f"R-Squared (R²)      : {r2:.3f}")
    print(f"Pearson Corr (PCC)  : {pearson_corr:.3f}")
    print("="*50)

if __name__ == "__main__":
    validate_brain_fast()