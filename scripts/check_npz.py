import numpy as np
import os
import matplotlib.pyplot as plt

# ================= CONFIG =================
NPZ_DIR = "npz_cache"
TARGET_LAYER = "L2"  # Change to L3, L4, etc. to check others
# ==========================================

# 1. Find a random file
files = [f for f in os.listdir(NPZ_DIR) if f.endswith(".npz")]
if not files:
    print(" No files found in npz_cache!")
    exit()

# Pick the first one (or change index to pick random)
test_file = files[0]
path = os.path.join(NPZ_DIR, test_file)

print(f" INSPECTING: {test_file}\n" + "="*40)

# 2. Load Data
data = np.load(path)
print(f" Keys found: {list(data.keys())}")

X = data["X"]
Y = data[f"Y_{TARGET_LAYER}"]

# 3. Check Shapes (CRITICAL: Must match!)
print(f"\n1️  SHAPE CHECK:")
print(f"   X shape: {X.shape}  (Samples, Features)")
print(f"   Y shape: {Y.shape}  (Samples,)")

if X.shape[0] != Y.shape[0]:
    print(" CRITICAL ERROR: X and Y have different number of rows!")
else:
    print(" Shapes are perfectly aligned.")

# 4. Check Values
print(f"\n2️  VALUE CHECK:")
print(f"   X (Input) Range:  {np.nanmin(X):.2f} to {np.nanmax(X):.2f}")
print(f"   Y (Target) Range: {np.nanmin(Y):.2f} to {np.nanmax(Y):.2f}")

# Check for NaNs
nan_x = np.isnan(X).sum()
nan_y = np.isnan(Y).sum()

print(f"   NaNs in X: {nan_x} (Should be 0)")
print(f"   NaNs in Y: {nan_y} (Some are okay, we filter them later)")

# 5. Physics Check (Scatter Plot)
# We plot Brightness Temp (Channel 0) vs Humidity
# If this looks like a random cloud, it's okay. If it's a straight line, it's bad.
if X.shape[0] > 0:
    plt.figure(figsize=(10, 5))
    
    # Filter valid Y for plotting
    mask = np.isfinite(Y)
    
    # Take a sample of 1000 points so plot is fast
    idx = np.random.choice(np.where(mask)[0], min(1000, mask.sum()), replace=False)
    
    plt.scatter(X[idx, 0], Y[idx], alpha=0.5, s=10, c='blue')
    plt.xlabel("Brightness Temp (Channel 1) [Kelvin/Scaled]")
    plt.ylabel(f"Relative Humidity ({TARGET_LAYER}) [%]")
    plt.title(f"Physics Check: TB vs RH for {test_file}")
    plt.grid(True, alpha=0.3)
    plt.show()
    print("\n Plot generated. Check if you see a 'cloud' of points.")
else:
    print("\n File is empty (0 samples). Try another file.")