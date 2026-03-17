import os
import re
import streamlit as st
import h5py
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
import tkinter as tk
from tkinter import filedialog

# ===================== CONFIG =====================
"""
Megha-Tropiques AI Humidity Retrieval
Streamlit interface for exploring SAPHIR satellite humidity retrieval.
"""

# Dataset location
# Priority:
# 1. Environment variable MEGATROPIQUES_DATA
# 2. Default local folder "data"
BASE_DIR = os.getenv("MEGATROPIQUES_DATA", "data")

# Model directory
MODEL_DIR = "models"

# Regex to extract orbit ID
ORBIT_RE = re.compile(r"_([0-9]{5})_")

# Ensure required folders exist
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
# ==================================================


# --- 1. MEMORY SETUP (SESSION STATE) ---
if "l1a_path" not in st.session_state:
    st.session_state.l1a_path = None

# --- 2. DYNAMIC MODEL CACHING ---
@st.cache_resource
def load_model(layer):
    model_path = os.path.join(MODEL_DIR, f"checkpoint_{layer}.txt")
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}. Have you finished training the {layer} layer yet?")
        st.stop()
    return lgb.Booster(model_file=model_path)

@st.cache_data
def build_l2a_map(base_dir):
    l2a_map = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if "L2A" in f and f.endswith(".h5"):
                match = ORBIT_RE.search(f)
                if match:
                    orbit_id = match.group(1)
                    l2a_map[orbit_id] = os.path.join(root, f)
    return l2a_map

L2A_FILE_MAP = build_l2a_map(BASE_DIR)

# --- 3. NATIVE FILE PICKER ---
def pick_file_native():
    root = tk.Tk()
    root.attributes('-topmost', True) 
    root.withdraw() 
    file_path = filedialog.askopenfilename(
        title="Select L1A HDF5 File",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

# --- 4. REVERSED METEOROLOGICAL COLOR SCALE ---
# Blue = Low Humidity (0%), Red/Orange = High Humidity (100%)
ISRO_COLORS = [
    [0.0, '#00008B'],   # Dark Blue (0% - Extremely Dry)
    [0.2, '#00FFFF'],   # Cyan (20%)
    [0.4, '#32CD32'],   # Lime Green (40%)
    [0.6, '#FFFF00'],   # Yellow (60%)
    [0.8, '#FFA500'],   # Orange (80%)
    [1.0, '#FF0000']    # Red (100% - Saturated/Rain)
]
BIAS_COLORS = "RdBu"

# --- 5. DYNAMIC DATA PROCESSING ---
def process_data(l1a_path, l2a_path, layer):
    with h5py.File(l1a_path, "r") as f:
        sd = f["ScienceData"]
        TB = np.stack([
            sd["TB_Samples_S2"][:], sd["TB_Samples_S3"][:],
            sd["TB_Samples_S4"][:], sd["TB_Samples_S5"][:]
        ], axis=-1) * 0.01
        lat1 = sd["Latitude_Samples"][:] * 0.01
        lon1 = sd["Longitude_Samples"][:] * 0.01
        inc = sd["IncidenceAngle_Samples"][:] * 0.01
        
        TBm = uniform_filter(TB, size=(3, 3, 1))
        mask_l1 = np.all(np.isfinite(TB), axis=-1) & (lat1 > -90)
        
    X_flat = np.column_stack([TB[mask_l1], TBm[mask_l1], lat1[mask_l1], lon1[mask_l1], inc[mask_l1]]).astype(np.float32)
    lats1 = lat1[mask_l1]
    lons1 = lon1[mask_l1]

    with h5py.File(l2a_path, "r") as f:
        sd = f["ScienceData"]
        lat2 = sd["Latitude"][:] * 0.01
        lon2 = sd["Longitude"][:] * 0.01
        rh_true_raw = sd[f"RelativeHumidity_{layer}"][:] * 0.01
        
    mask_l2 = (lat2 > -90) & (lat2 < 90)
    lats2 = lat2[mask_l2]
    lons2 = lon2[mask_l2]
    rh_true_clean = rh_true_raw[mask_l2]

    return X_flat, lats1, lons1, rh_true_clean, lats2, lons2

# --- 6. METRICS CALCULATOR ---
def calculate_metrics(y_true, y_pred):
    n_pixels = len(y_true)
    bias_arr = y_pred - y_true
    
    mean_bias = np.mean(bias_arr)
    mae = np.mean(np.abs(bias_arr))
    mse = np.mean(bias_arr**2)
    rmse = np.sqrt(mse)
    
    r = np.corrcoef(y_true, y_pred)[0, 1]
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    if std_true == 0 or mean_true == 0:
        kge = np.nan
    else:
        alpha = std_pred / std_true
        beta = mean_pred / mean_true
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        
    return n_pixels, rmse, mean_bias, mse, mae, kge

# --- 7. PLOTTING FUNCTIONS ---

def plot_swath(lats, lons, values, title, colorscale, vmin, vmax):
    stride = 4 
    fig = go.Figure(go.Scattergeo(
        lon = lons[::stride], lat = lats[::stride],
        marker = dict(
            color = values[::stride], colorscale = colorscale,
            cmin = vmin, cmax = vmax,
            colorbar_title = "RH %" if "Bias" not in title else "Error %",
            size = 2.5, opacity=1.0
        ),
        mode = 'markers', hoverinfo='skip'  
    ))
    fig.update_layout(
        title = dict(text=title, font=dict(size=24)),
        geo = dict(
            showcoastlines=True, coastlinecolor="Black",
            showland=True, landcolor="#e5e7eb",    
            showocean=True, oceancolor="#f0f8ff",  
            projection_type="equirectangular", fitbounds="locations", 
        ),
        margin={"r":20,"t":60,"l":20,"b":20}, height=650  
    )
    return fig

def plot_scatter(y_true, y_pred, layer):
    step = max(1, len(y_true) // 100000)
    yt, yp = y_true[::step], y_pred[::step]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yt, y=yp, mode='markers', marker=dict(color='blue', size=2, opacity=0.15), name="Predictions", hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', line=dict(color='red', width=3, dash='dash'), name="Perfect Accuracy", hoverinfo='skip'))
    fig.update_layout(
        title = dict(text=f"True vs Predicted Correlation - {layer}", font=dict(size=20)),
        xaxis_title="True RH (%)", yaxis_title="Predicted RH (%)",
        xaxis=dict(range=[0, 100]), yaxis=dict(range=[0, 100]),
        height=500, showlegend=False, plot_bgcolor="white" 
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    return fig

# THE BULLETPROOF DENSITY FIX: Using pure NumPy to eliminate the purple background
def plot_density(y_true, y_pred, layer):
    hist, x_edges, y_edges = np.histogram2d(y_true, y_pred, bins=100, range=[[0, 100], [0, 100]])
    hist = hist.T 
    hist[hist == 0] = np.nan # This forces all empty space to be completely invisible!

    fig = go.Figure(go.Heatmap(
        z=hist,
        x=(x_edges[:-1] + x_edges[1:]) / 2,
        y=(y_edges[:-1] + y_edges[1:]) / 2,
        colorscale='Viridis',
        colorbar=dict(title='Pixel Count'),
        hoverinfo='x+y+z'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], 
        mode='lines', 
        line=dict(color='red', width=2, dash='dash'), 
        name="1:1 Line", hoverinfo='skip'
    ))
    
    fig.update_layout(
        title = dict(text=f"Data Density - {layer}", font=dict(size=20)),
        xaxis_title="True RH (%)", yaxis_title="Predicted RH (%)",
        xaxis=dict(range=[0, 100]), yaxis=dict(range=[0, 100]),
        height=500, showlegend=False, 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)' # Transparent!
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')
    return fig

def plot_bias_histogram(bias, layer):
    counts, bin_edges = np.histogram(bias, bins=100, range=(-40, 40))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure(go.Bar(
        x=bin_centers,
        y=counts,
        marker=dict(
            color=bin_centers,       
            colorscale=BIAS_COLORS,  
            cmin=-30,
            cmax=30,
            line=dict(width=0),      
            showscale=True,          
            colorbar=dict(title="Bias Error %")
        )
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="white", line_width=2) 
    fig.update_layout(
        title = dict(text=f"Bias Distribution (Colored) - {layer}", font=dict(size=20)),
        xaxis_title="Bias (Predicted - True) %", 
        yaxis_title="Number of Pixels",
        xaxis=dict(range=[-40, 40]), 
        height=500, 
        bargap=0,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)' # Transparent!
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')
    return fig


# ===================== GUI LAYOUT =====================
st.title(" Mega-Tropiques AI: Humidity Layer Simulator")
st.markdown("Select an L1A file and choose your desired atmospheric layer to map.")

col1, col2 = st.columns([1, 2])
with col1:
    selected_layer = st.selectbox(" Select Atmospheric Layer:", ["L2", "L3", "L4", "L5"])

with col2:
    st.write("") 
    st.write("")
    if st.button("📁 Browse & Select L1A File", type="primary"):
        selected_file = pick_file_native()
        if selected_file:
            st.session_state.l1a_path = selected_file
            st.rerun()

if st.session_state.l1a_path:
    l1a_path = st.session_state.l1a_path
    st.success(f"**Selected L1A File:** `{l1a_path}`")
    
    filename = os.path.basename(l1a_path)
    match = ORBIT_RE.search(filename)
    
    if not match:
        st.error("❌ Could not detect a valid 5-digit Orbit ID in the filename.")
        st.stop()
        
    orbit_id = match.group(1)
    l2a_path = L2A_FILE_MAP.get(orbit_id)
            
    if not l2a_path:
        st.error(f"❌ Could not find a matching L2A file for orbit {orbit_id} anywhere inside {BASE_DIR}.")
        st.stop()
        
    st.success(f"**Auto-Found L2A File:** `{l2a_path}`")
    st.markdown("---")
    
    if st.button(f" Generate {selected_layer} Swath Maps & Analytics"):
        with st.spinner(f"Loading {selected_layer} Model and computing physics..."):
            
            model = load_model(selected_layer)
            X_data, l1_lats, l1_lons, y_true_raw, l2_lats, l2_lons = process_data(l1a_path, l2a_path, selected_layer)
            y_pred = model.predict(X_data)
            
            tree = cKDTree(np.column_stack([l2_lats, l2_lons]))
            dist, idx = tree.query(np.column_stack([l1_lats, l1_lons]), k=1)
            valid_match = dist < 0.05
            
            lats_final = l1_lats[valid_match]
            lons_final = l1_lons[valid_match]
            y_pred_final = y_pred[valid_match]
            y_true_final = y_true_raw[idx[valid_match]]
            
            valid_rh = (y_true_final >= 0) & (y_true_final <= 100)
            lats_final = lats_final[valid_rh]
            lons_final = lons_final[valid_rh]
            y_pred_final = y_pred_final[valid_rh]
            y_true_final = y_true_final[valid_rh]
            bias = y_pred_final - y_true_final
            
            # --- CALCULATE METRICS ---
            n_pix, rmse, mbias, mse, mae, kge = calculate_metrics(y_true_final, y_pred_final)

            st.markdown(f"###  Scientific Metrics for {selected_layer}")
            
            # --- DISPLAY METRICS ---
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Total Pixels", f"{n_pix:,}")
            m2.metric("RMSE", f"{rmse:.2f} %")
            m3.metric("Mean Bias", f"{mbias:.2f} %")
            m4.metric("MSE", f"{mse:.2f}")
            m5.metric("MAE", f"{mae:.2f} %")
            m6.metric("KGE Score", f"{kge:.3f}")
            
            st.markdown("---")
            st.markdown("###  Error Distribution & Correlations")
            
            st.plotly_chart(plot_scatter(y_true_final, y_pred_final, selected_layer), use_container_width=True, config={'scrollZoom': False})
            
            # DENSITY AND BIAS (Transparent backgrounds!)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_density(y_true_final, y_pred_final, selected_layer), use_container_width=True, config={'scrollZoom': False})
            with c2:
                st.plotly_chart(plot_bias_histogram(bias, selected_layer), use_container_width=True, config={'scrollZoom': False})

            st.markdown("---")
            st.markdown("###  Geospatial Rendering")
            
            st.plotly_chart(plot_swath(lats_final, lons_final, y_true_final, f"Actual RH Map - {selected_layer} (Ground Truth)", ISRO_COLORS, 0, 100), use_container_width=True, config={'scrollZoom': False})
            st.plotly_chart(plot_swath(lats_final, lons_final, y_pred_final, f"Simulated RH Map - {selected_layer} (AI Output)", ISRO_COLORS, 0, 100), use_container_width=True, config={'scrollZoom': False})
            st.plotly_chart(plot_swath(lats_final, lons_final, bias, f"Bias Map - {selected_layer} (Predicted - Actual)", BIAS_COLORS, -30, 30), use_container_width=True, config={'scrollZoom': False})