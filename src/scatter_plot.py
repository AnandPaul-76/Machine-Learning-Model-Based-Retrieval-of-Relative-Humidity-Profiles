import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ============== CONFIG =================
NPZ_DIR = "npz_cache"
MODEL_FILE = "models/RH_L2.pkl"   # or RH_L2_FINAL.pkl if you renamed
MAX_POINTS = 300_000              # safe for laptop
SEED = 42

np.random.seed(SEED)

# ============== LOAD DATA ==============
Xs, ys = [], []
files = sorted(os.listdir(NPZ_DIR))

for f in files:
    data = np.load(os.path.join(NPZ_DIR, f))
    X = data["X"]
    y = data["Y_L2"]

    if X.shape[0] == 0:
        continue

    # sample per orbit
    take = min(5_000, X.shape[0])
    idx = np.random.choice(X.shape[0], take, replace=False)
    Xs.append(X[idx])
    ys.append(y[idx])

    if sum(x.shape[0] for x in Xs) >= MAX_POINTS:
        break

X_all = np.vstack(Xs)
y_true = np.hstack(ys)

print("Scatter samples:", X_all.shape[0])

# ============== PREDICT =================
model = joblib.load(MODEL_FILE)
y_pred = model.predict(X_all)

# ============== SCATTER PLOT ===========
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, s=1, alpha=0.3)
plt.plot([0, 100], [0, 100], 'r--')  # perfect prediction line

plt.xlabel("Actual RH (%)")
plt.ylabel("Predicted RH (%)")
plt.title("Actual vs Predicted RH (L2)")
plt.grid(True)

plt.tight_layout()
plt.show()
