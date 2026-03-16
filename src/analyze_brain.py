import os
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go

# ================= CONFIG =================
MODEL_DIR = "models"
LAYERS = ["L2", "L3", "L4", "L5"]

# The exact 11 inputs fed into the AI
FEATURES = [
    "TB Channel 2", "TB Channel 3", "TB Channel 4", "TB Channel 5",
    "Smoothed Mean Ch 2", "Smoothed Mean Ch 3", "Smoothed Mean Ch 4", "Smoothed Mean Ch 5",
    "Latitude", "Longitude", "Incidence Angle"
]
# ==========================================

def plot_all_brains():
    print(" Booting up the Deep Brain Analyzer...")
    
    for layer in LAYERS:
        model_path = os.path.join(MODEL_DIR, f"checkpoint_{layer}.txt")
        
        if not os.path.exists(model_path):
            print(f" Could not find model for {layer}. Skipping...")
            continue
            
        print(f" Extracting physics logic from Layer {layer}...")
        model = lgb.Booster(model_file=model_path)
        
        # Calculate how much error each feature mathematically removed (Information Gain)
        importances = model.feature_importance(importance_type='gain') 
        
        # Sort them from lowest to highest for the bar chart
        indices = np.argsort(importances)
        sorted_features = [FEATURES[i] for i in indices]
        sorted_importances = importances[indices]
        
        # Draw the chart
        fig = go.Figure(go.Bar(
            x=sorted_importances,
            y=sorted_features,
            orientation='h',
            marker=dict(
                color=sorted_importances,
                colorscale='Viridis', # Beautiful scientific color gradient
            )
        ))
        
        fig.update_layout(
            title=dict(text=f"AI Brain Analysis: Physics Feature Importance for Layer {layer}", font=dict(size=24)),
            xaxis_title="Relative Importance (Information Gain)",
            yaxis_title="Satellite Inputs",
            height=600,
            margin={"l": 150}, # Extra space for the labels
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        
        # Automatically pop it open in your web browser!
        fig.show()
        
    print(" All brain scans complete! Check your web browser tabs.")

if __name__ == "__main__":
    plot_all_brains()