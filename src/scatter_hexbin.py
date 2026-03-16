import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ============== CONFIG =================
NPZ_DIR = "npz_cache"
MODEL_FILE = "models/RH_L2.pkl"   # or RH_L2_FINAL.pkl
MAX_SAMPLES_PER_ORBIT = 10_000    # 10k × 3200 = 32M max
MAX_TOTAL = 5_000_000             # cap for laptop safety
SEED = 42

np.random.seed(SEED)

# ============== LOAD DATA ==============
Xs, ys = [], []
files = sorted(os.listdir(NPZ_DIR))

total = 0
for f in files:
    data = np.load(os.path.join(NPZ_DIR, f))
    X = data["X"]
    y = data["Y_L2"]

    if X.shape[0] == 0:
        continue

    take = min(MAX_SAMPLES_PER_ORBIT, X.shape[0])
    idx = np.random.choice(X.shape[0], take, replace=False)

    Xs.append(X[idx])
    ys.append(y[idx])
    total += take

    if total >= MAX_TOTAL:
        break

X_all = np.vstack(Xs)
y_true = np.hstack(ys)

print("Total points used:", X_all.shape[0])

# ============== PREDICT =================
model = joblib.load(MODEL_FILE)
y_pred = model.predict(X_all)

# ============== HEXBIN PLOT =============
plt.figure(figsize=(7, 6))
hb = plt.hexbin(
    y_true,
    y_pred,
    gridsize=150,
    cmap="viridis",
    mincnt=1
)

plt.plot([0, 100], [0, 100], 'r--', lw=1)
plt.xlabel("Actual RH (%)")
plt.ylabel("Predicted RH (%)")
plt.title("Actual vs Predicted RH (L2) — Density Plot")
plt.colorbar(hb, label="Sample density")
plt.tight_layout()
plt.show()
