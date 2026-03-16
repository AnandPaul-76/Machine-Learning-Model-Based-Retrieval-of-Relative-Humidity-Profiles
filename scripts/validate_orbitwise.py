import os
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error

# ================= CONFIG =================
NPZ_DIR = "npz_cache"
MODEL_FILE = "models/RH_L2.pkl"

TEST_FRACTION = 0.2            # 20% orbits completely unseen
MAX_SAMPLES_PER_ORBIT = 50_000 # keep it fair & fast
SEED = 42
# ==========================================

# ================ LOAD FILES ================
files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")])

np.random.seed(SEED)
np.random.shuffle(files)

n_test = int(len(files) * TEST_FRACTION)
test_files = files[:n_test]

Xs, ys = [], []

for f in test_files:
    data = np.load(os.path.join(NPZ_DIR, f))
    X = data["X"]
    y = data["Y_L2"]

    if X.shape[0] == 0:
        continue

    # Filter invalid humidity values
    valid = np.isfinite(y)
    X = X[valid]
    y = y[valid]

    if X.shape[0] == 0:
        continue

    # Limit samples per orbit
    if X.shape[0] > MAX_SAMPLES_PER_ORBIT:
        idx = np.random.choice(X.shape[0], MAX_SAMPLES_PER_ORBIT, replace=False)
        X, y = X[idx], y[idx]

    Xs.append(X)
    ys.append(y)

# Combine
X_test = np.vstack(Xs)
y_test = np.hstack(ys)

print("Unseen orbits used:", len(test_files))
print("Validation samples:", X_test.shape[0])

# ================ PREDICT ====================
model = joblib.load(MODEL_FILE)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)

# ================ JUDGEMENT ==================
print("\nORBIT-WISE MAE (%RH):", round(mae, 3))

if mae <= 4.0:
    verdict = "EXTREMELY GOOD (research-grade, near physical limit)"
elif mae <= 5.0:
    verdict = "VERY GOOD (strong generalization)"
elif mae <= 6.0:
    verdict = "OK / ACCEPTABLE (can be improved)"
else:
    verdict = "NEEDS WORK (check features / preprocessing)"

print("MODEL VERDICT:", verdict)