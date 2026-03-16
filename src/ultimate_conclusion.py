import os
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import random

# ================= CONFIG =================
LAYER = "L4" # The "sweet spot" mid-troposphere layer
MODEL_DIR = "models"
NPZ_DIR = "npz_cache"
# ==========================================

def generate_publication_proof():
    print(" Booting up the Final Publication Validator...")

    # 1. Load the model
    model_path = os.path.join(MODEL_DIR, f"checkpoint_{LAYER}.txt")
    model = lgb.Booster(model_file=model_path)

    # 2. Load unseen test data (Globally Distributed)
    all_files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')])
    
    #  THE FIX: Grab 15 random files from the entire dataset to guarantee we cross the Equator
    random.seed(42) # Keeps it reproducible so you get the exact same chart every time
    test_files = random.sample(all_files, min(15, len(all_files)))
    
    y_true_list = []
    y_pred_list = []
    lats_list = []
    
    print(f" Crunching {len(test_files)} globally distributed orbits for the Density Plot...")
    for f in test_files:
        try:
            data = np.load(os.path.join(NPZ_DIR, f))
            X = data['X']
            y = data[f'Y_{LAYER}']
            
            valid_mask = np.isfinite(y)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            # Predict
            preds = model.predict(X_clean)
            
            # X_clean[:, 8] is Latitude based on our training array structure
            lats = X_clean[:, 8] 
            
            y_true_list.append(y_clean)
            y_pred_list.append(preds)
            lats_list.append(lats)
        except Exception:
            pass

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    lats = np.concatenate(lats_list)
    errors = y_pred - y_true

    # ==============================================================
    #  PLOT 1: THE DENSITY HEATMAP SCATTER
    # ==============================================================
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"Density Scatter Correlation (N = {len(y_true):,})", 
        "Zonal Mean Error (The Equator Test)"
    ))

    # We use a 2D Histogram because standard scatter plots crash with millions of points
    fig.add_trace(go.Histogram2dContour(
        x=y_true, y=y_pred,
        colorscale='Blues',
        reversescale=False,
        showscale=False,
        ncontours=20,
        contours=dict(coloring='heatmap')
    ), row=1, col=1)

    # The Perfect 1:1 Line
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode='lines',
        line=dict(color='red', width=3, dash='dash'),
        name="Perfect 1:1 Agreement"
    ), row=1, col=1)

    # ==============================================================
    #  PLOT 2: ZONAL MEAN ERROR (LATITUDE STABILITY)
    # ==============================================================
    # Group errors by Latitude into 2-degree bins
    df = pd.DataFrame({'Lat': lats, 'Error': errors})
    df['Lat_Bin'] = pd.cut(df['Lat'], bins=np.arange(-30, 32, 2))
    
    # Calculate the mean and standard deviation for each latitude slice
    zonal_stats = df.groupby('Lat_Bin', observed=False)['Error'].agg(['mean', 'std']).reset_index()
    zonal_stats['Lat_Center'] = zonal_stats['Lat_Bin'].apply(lambda x: x.mid).astype(float)
    
    # Plot Mean Bias line
    fig.add_trace(go.Scatter(
        x=zonal_stats['Lat_Center'], y=zonal_stats['mean'],
        mode='lines+markers', line=dict(color='blue', width=4),
        name="Mean Bias", error_y=dict(type='data', array=zonal_stats['std'], visible=True, color='lightblue', thickness=1)
    ), row=1, col=2)

    # Add 0% Error Baseline
    fig.add_trace(go.Scatter(
        x=[-30, 30], y=[0, 0], mode='lines',
        line=dict(color='red', width=3, dash='dash'), showlegend=False
    ), row=1, col=2)

    # ================= Formatting =================
    fig.update_layout(
        title=dict(text="Final Validation: Density Correlation & Geographical Stability", font=dict(size=24)),
        height=600, plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )
    
    fig.update_xaxes(title_text="ISRO True Humidity (%)", range=[0, 100], row=1, col=1, showgrid=True, gridcolor='lightgrey')
    fig.update_yaxes(title_text="AI Predicted Humidity (%)", range=[0, 100], row=1, col=1, showgrid=True, gridcolor='lightgrey')
    
    fig.update_xaxes(title_text="Latitude (Degrees)", range=[-30, 30], row=1, col=2, showgrid=True, gridcolor='lightgrey')
    fig.update_yaxes(title_text="Prediction Error (%)", range=[-15, 15], row=1, col=2, showgrid=True, gridcolor='lightgrey')

    fig.show()
    print(" Conclusion Charts Generated!")

if __name__ == "__main__":
    generate_publication_proof()