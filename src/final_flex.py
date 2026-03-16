import os
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================= CONFIG =================
LAYERS = ["L5", "L4", "L3", "L2"] # Ordered from High Altitude (L5) to Low Altitude (L2)
MODEL_DIR = "models"
NPZ_DIR = "npz_cache"
# ==========================================

def run_the_final_flex():
    print(" Launching Virtual Weather Balloons & Computing Error Physics...")

    # 1. Load the brains
    models = {}
    for layer in LAYERS:
        models[layer] = lgb.Booster(model_file=os.path.join(MODEL_DIR, f"checkpoint_{layer}.txt"))

    # 2. Find a "complete" unseen test file that has all 4 layers inside it
    all_files = sorted([f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')])
    data = None
    
    for f in reversed(all_files):
        try:
            temp_data = np.load(os.path.join(NPZ_DIR, f))
            # Check if this specific file has L2, L3, L4, and L5 saved inside
            if all(f'Y_{layer}' in temp_data.files for layer in LAYERS):
                data = temp_data
                print(f" Found a complete 3D atmospheric file: {f}")
                break
        except Exception:
            continue
            
    if data is None:
        print(" Could not find a file with all 4 layers. Did you delete the cache?")
        return
    
    X = data['X']
    
    # 3. Find pixels where we have valid data for ALL 4 layers
    valid_mask = np.ones(X.shape[0], dtype=bool)
    for layer in LAYERS:
        valid_mask &= np.isfinite(data[f'Y_{layer}'])
        
    X_clean = X[valid_mask]
    
    print(f" Found {X_clean.shape[0]:,} valid vertical columns to test.")

    # 4. Predict the entire 3D atmosphere!
    y_true_all = {layer: data[f'Y_{layer}'][valid_mask] for layer in LAYERS}
    y_pred_all = {layer: models[layer].predict(X_clean) for layer in LAYERS}

    # ==============================================================
    #  FLEX 1: THE VERTICAL ATMOSPHERIC PROFILES (WEATHER BALLOONS)
    # ==============================================================
    l2_truth = y_true_all["L2"]
    
    dry_idx = np.argmin(np.abs(l2_truth - 20))    # ~20% Humidity (Dry)
    norm_idx = np.argmin(np.abs(l2_truth - 50))   # ~50% Humidity (Average)
    storm_idx = np.argmin(np.abs(l2_truth - 95))  # ~95% Humidity (Storm/Rain)
    
    pixels_to_plot = [
        ("Desert/Dry Atmosphere", dry_idx),
        ("Standard Atmosphere", norm_idx),
        ("Deep Storm System", storm_idx)
    ]

    fig_prof = make_subplots(rows=1, cols=3, subplot_titles=[p[0] for p in pixels_to_plot])

    for col, (title, idx) in enumerate(pixels_to_plot):
        true_profile = [y_true_all[layer][idx] for layer in LAYERS]
        pred_profile = [y_pred_all[layer][idx] for layer in LAYERS]
        
        # True Satellite Line (Solid Black)
        fig_prof.add_trace(go.Scatter(
            x=true_profile, y=LAYERS, mode='lines+markers',
            name='ISRO Satellite (Truth)', line=dict(color='black', width=4),
            marker=dict(size=10), showlegend=(col==0)
        ), row=1, col=col+1)
        
        # AI Prediction Line (Dashed Red)
        fig_prof.add_trace(go.Scatter(
            x=pred_profile, y=LAYERS, mode='lines+markers',
            name='AI Emulator (Prediction)', line=dict(color='red', width=4, dash='dash'),
            marker=dict(size=10, symbol='x'), showlegend=(col==0)
        ), row=1, col=col+1)
        
        fig_prof.update_xaxes(title_text="Relative Humidity (%)", range=[0, 100], row=1, col=col+1)

    fig_prof.update_yaxes(title_text="Atmospheric Layer (Altitude)", row=1, col=1)
    fig_prof.update_layout(
        title=dict(text="3D Vertical Sounding: AI vs. True Atmospheric Physics", font=dict(size=24)),
        height=600, plot_bgcolor="white", paper_bgcolor="white"
    )
    fig_prof.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig_prof.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig_prof.show()

    # ==============================================================
    #  FLEX 2: THE GAUSSIAN ERROR BELL CURVE (PDF)
    # ==============================================================
    errors = y_pred_all["L4"] - y_true_all["L4"]
    errors_plot = errors[::10] 

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=errors_plot, nbinsx=150, 
        marker_color='#1f77b4', opacity=0.75,
        name="Error Distribution"
    ))
    
    fig_hist.add_vline(x=0, line_width=3, line_dash="dash", line_color="red", 
                       annotation_text="Perfect Accuracy (0% Error)")

    fig_hist.update_layout(
        title=dict(text="Error Distribution (Probability Density Function) - L4", font=dict(size=24)),
        xaxis_title="Prediction Error (Predicted - Actual %)",
        yaxis_title="Number of Pixels",
        xaxis=dict(range=[-25, 25]), 
        height=600, plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )
    fig_hist.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig_hist.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig_hist.show()

    print(" Boom. Mind-blowing charts opened in your browser.")

if __name__ == "__main__":
    run_the_final_flex()