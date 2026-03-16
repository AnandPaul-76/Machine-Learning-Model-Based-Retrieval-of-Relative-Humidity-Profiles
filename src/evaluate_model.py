import os
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================= CONFIG =================
RH_LAYER = "L2"  # The layer you trained
CKPT_PATH = f"models/checkpoint_{RH_LAYER}.txt"
NPZ_DIR = "npz_cache"
# ==========================================

def evaluate():
    if not os.path.exists(CKPT_PATH):
        print(f" Error: Could not find model at {CKPT_PATH}")
        return

    print(" Loading Model (Brain)...")
    model = lgb.Booster(model_file=CKPT_PATH)

    # 1. Get an UNSEEN file (the very last file in your folder)
    files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])
    test_file = files[-1]  # The last file (definitely not in the first 69 batches)
    print(f" Loading Unseen Test File: {test_file}")

    data = np.load(os.path.join(NPZ_DIR, test_file))
    X = data["X"]
    y_true = data[f"Y_{RH_LAYER}"]

    # 2. Clean Data (Remove NaNs)
    valid = np.isfinite(y_true)
    X_test = X[valid]
    y_true = y_true[valid]

    if len(y_true) == 0:
        print(" No valid pixels in this test file. Try another one.")
        return

    print(f" Testing on {len(y_true):,} pixels...")

    # 3. Make AI Predictions
    y_pred = model.predict(X_test)

    # 4. Calculate Hard Error Rates
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n" + "="*40)
    print(" ERROR RATES & ACCURACY SCORE")
    print("="*40)
    print(f"Mean Absolute Error (MAE) : {mae:.2f}% (How far off the AI is on average)")
    print(f"Root Mean Sq Error (RMSE) : {rmse:.2f}% (Penalizes massive errors)")
    print(f"R-Squared (R²) Score      : {r2:.3f} (1.0 is perfect, >0.7 is excellent)")
    print("="*40)

    # 5. Extract Latitude (Index 8) and Longitude (Index 9) for Mapping
    # Based on your preprocessing script: 0-3=TB, 4-7=TBm, 8=Lat, 9=Lon, 10=Inc
    lats = X_test[:, 8]
    lons = X_test[:, 9]

    # 6. Plotting the Visual Proof (Physics!)
    print("\n Generating Visual Proof Maps... (A window will pop up)")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: True Humidity
    sc1 = ax1.scatter(lons, lats, c=y_true, cmap="turbo", s=1, vmin=0, vmax=100)
    ax1.set_title("Actual Satellite Humidity (Truth)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    plt.colorbar(sc1, ax=ax1, label="Relative Humidity %")

    # Plot 2: AI Predicted Humidity
    sc2 = ax2.scatter(lons, lats, c=y_pred, cmap="turbo", s=1, vmin=0, vmax=100)
    ax2.set_title("AI Predicted Humidity (Did it learn physics?)")
    ax2.set_xlabel("Longitude")
    
    # Plot 3: Scatter Correlation
    ax3.scatter(y_true, y_pred, alpha=0.1, s=1, color="blue")
    ax3.plot([0, 100], [0, 100], color="red", linestyle="--") # Perfect line
    ax3.set_title("True vs Predicted Correlation")
    ax3.set_xlabel("True Humidity")
    ax3.set_ylabel("Predicted Humidity")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()