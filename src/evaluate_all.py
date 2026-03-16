import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import time

# ================= CONFIG =================
LAYERS = ["L2", "L3", "L4", "L5"]
MODEL_DIR = "models"
NPZ_DIR = "npz_cache"
FILES_TO_TEST = 10  # Tests roughly 7.5 Million unseen pixels!
# ==========================================

def generate_master_table():
    print(" Booting up the Master Evaluation Script...")
    
    all_files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')])
    test_files = all_files[-FILES_TO_TEST:]
    
    print(f" Loading {len(test_files)} unseen test files for final validation...\n")
    
    results = {}

    for layer in LAYERS:
        model_path = os.path.join(MODEL_DIR, f"checkpoint_{layer}.txt")
        if not os.path.exists(model_path):
            print(f" Missing model for {layer}! Skipping...")
            continue
            
        print(f" Testing Layer {layer}...", end=" ")
        start_time = time.time()
        model = lgb.Booster(model_file=model_path)
        
        y_true = []
        y_pred = []
        
        for f in test_files:
            try:
                data = np.load(os.path.join(NPZ_DIR, f))
                X = data['X']
                y = data[f'Y_{layer}']
                
                valid = np.isfinite(y)
                X_clean = X[valid]
                y_clean = y[valid]
                
                if len(y_clean) > 0:
                    preds = model.predict(X_clean)
                    y_true.append(y_clean)
                    y_pred.append(preds)
            except Exception:
                pass
        
        if not y_true:
            print("Failed (No data).")
            continue
            
        # Fast RAM-friendly concatenation
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        
        # Calculate the core scientific metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = np.mean(y_pred - y_true)
        r2 = r2_score(y_true, y_pred)
        pcc, _ = pearsonr(y_true, y_pred)
        
        results[layer] = {
            "MAE": mae,
            "RMSE": rmse,
            "Bias": bias,
            "R2": r2,
            "PCC": pcc
        }
        print(f"Done! ({len(y_true):,} pixels in {time.time()-start_time:.1f}s)")

    # --- PRINT THE PUBLICATION TABLE ---
    print("\n" + "="*85)
    print(" FINAL MASTER EVALUATION TABLE ")
    print("="*85)
    print(f"| {'Layer':<6} | {'MAE (%)':<10} | {'RMSE (%)':<10} | {'Bias (%)':<10} | {'R-Squared':<10} | {'Pearson (PCC)':<13} |")
    print("|" + "-"*8 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*15 + "|")
    
    for layer in LAYERS:
        if layer in results:
            r = results[layer]
            print(f"| {layer:<6} | {r['MAE']:<10.3f} | {r['RMSE']:<10.3f} | {r['Bias']:<10.3f} | {r['R2']:<10.3f} | {r['PCC']:<13.3f} |")
    
    print("="*85)
    print(" Ready for the 'Results' section of your manuscript!")

if __name__ == "__main__":
    generate_master_table()