import os
import numpy as np
import joblib
import pandas as pd

# ================= CONFIG =================
NPZ_DIR = "npz_cache"
MODEL_FILE = "models/RH_L2.pkl"   # use your frozen model
MAX_POINTS = 20                         # how many rows to print
SEED = 42

np.random.seed(SEED)

# ================= LOAD ONE ORBIT =================
files = sorted(os.listdir(NPZ_DIR))
f = files[np.random.randint(len(files))]   # random unseen-ish orbit

data = np.load(os.path.join(NPZ_DIR, f))
X = data["X"]
y_true = data["Y_L2"]

# sample a few points
idx = np.random.choice(len(y_true), MAX_POINTS, replace=False)
X_s = X[idx]
y_s = y_true[idx]

# ================= PREDICT =================
model = joblib.load(MODEL_FILE)
y_pred = model.predict(X_s)

# ================= DISPLAY =================
df = pd.DataFrame({
    "Actual_RH": y_s,
    "Predicted_RH": y_pred,
    "Abs_Error": np.abs(y_s - y_pred)
})

print(f"\nShowing {MAX_POINTS} samples from orbit {f}\n")
print(df.round(2))
