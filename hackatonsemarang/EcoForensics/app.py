import streamlit as st
import folium
from streamlit_folium import st_folium
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import requests
from pathlib import Path
import segmentation_models_pytorch as smp
import sys
import os

# --- 1. SETUP PATH & IMPORT ---
current_file_path = Path(__file__).resolve()
scripts_folder = current_file_path.parent / "scripts"
if str(scripts_folder) not in sys.path:
    sys.path.append(str(scripts_folder))

try:
    from gee_loader import get_gee_data
except ImportError:
    try:
        from scripts.gee_loader import get_gee_data
    except ImportError:
        st.error("❌ File 'gee_loader.py' tidak ditemukan!")
        st.stop()

# --- 2. CONFIG HALAMAN ---
st.set_page_config(
    page_title="Eco-Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CSS SUPER CLEAN (SAFE MODE) ---
st.markdown("""
<style>
    /* 1. RESET GLOBAL */
    .stApp {
        background-color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* 2. HEADER TRANSPARAN (Agar Sidebar tidak hilang) */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 100;
    }
    button[data-testid="baseButton-header"] {
        color: #000000 !important;
    }
    .st-emotion-cache-12fmw14, .stDeployButton { display: none; }

    /* 3. SIDEBAR CLEAN */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
        padding-top: 1rem;
    }
    
    /* 4. MAP CONTAINER STYLE */
    .block-container {
        padding-top: 2rem !important; 
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Style Iframe Peta */
    iframe {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #f1f5f9;
    }

    /* 5. TYPOGRAPHY */
    h1, h2, h3, p, label, span, div {
        color: #0f172a !important;
    }
    
    /* 6. INPUT FIELDS */
    .stTextInput input, .stNumberInput input {
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
        border-radius: 8px;
    }
    
    /* 7. RESULT CARDS */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-radius: 10px;
        padding: 15px;
    }
    
    /* 8. BUTTON STYLE */
    div.stButton > button:first-child {
        background-color: #000000;
        color: #ffffff !important; /* DIPASTIKAN PUTIH */
        border-radius: 8px;
        border: none;
        height: 3rem;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. SESSION STATE ---
if 'lat_default' not in st.session_state: st.session_state['lat_default'] = -6.8946
if 'lon_default' not in st.session_state: st.session_state['lon_default'] = 110.6401
if 'analysis_result' not in st.session_state: st.session_state['analysis_result'] = None

# --- 5. LOAD MODEL ---
@st.cache_resource
def load_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = Path("models/sar_hydra_v2.pth") 
    
    model = smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=4, classes=1).to(DEVICE)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model, DEVICE
    else:
        return None, DEVICE

model, DEVICE = load_model()

# --- 6. HELPERS ---
def pad_image(array, divisor=32):
    c, h, w = array.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h == 0 and pad_w == 0: return array
    return np.pad(array, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

def preprocess_anti_crash(s1_path, dem_path):
    with rasterio.open(s1_path) as src:
        bands = src.read()
        if bands.shape[0] < 2:
            s1 = np.concatenate([bands, bands], axis=0)
        else:
            s1 = bands[:2]
        s1 = (np.clip(s1, -30, 0) + 30) / 30.0

    with rasterio.open(dem_path) as src:
        dem = np.clip(src.read(1), 0, 1000) / 1000.0

    target_h, target_w = s1.shape[1], s1.shape[2]
    if dem.shape != s1.shape[1:]:
        import cv2
        dem = cv2.resize(dem, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    input_stack = np.stack([s1[0], s1[1], dem, s1[0]])
    tensor = torch.from_numpy(pad_image(input_stack)).unsqueeze(0).float()
    return tensor, s1.shape[1], s1.shape[2]

def search_nominatim(query):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': query, 'format': 'json', 'limit': 1}
        headers = {'User-Agent': 'EcoForensicsApp/1.0'}
        r = requests.get(url, params=params, headers=headers, timeout=5)
        data = r.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except:
        pass
    return None, None

# --- 7. SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='font-size: 1.8rem; font-weight: 900; margin-bottom: 0px;'>Eco-Forensics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; color: #64748b; margin-bottom: 30px;'>AI-Driven Hydrology Audit</p>", unsafe_allow_html=True)
    
    # SEARCH
    st.markdown("**1. SEARCH LOCATION**")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Area", placeholder="Ex: Semarang", label_visibility="collapsed")
    with col2:
        btn_cari = st.button("🔍")

    if btn_cari and search_query:
        slat, slon = search_nominatim(search_query)
        if slat:
            st.session_state['lat_default'] = slat
            st.session_state['lon_default'] = slon
            st.session_state['analysis_result'] = None
            st.rerun()
        else:
            st.error("Not Found")

    st.markdown("<br>", unsafe_allow_html=True)

    # COORDS
    st.markdown("**2. TARGET COORDINATES**")
    lat_input = st.number_input("Latitude", value=st.session_state['lat_default'], format="%.4f")
    lon_input = st.number_input("Longitude", value=st.session_state['lon_default'], format="%.4f")
    
    st.markdown("---")
    
    # ACTION
    analyze_btn = st.button("🚀 START AUDIT")
    
    st.markdown("<br><br><small style='color:#94a3b8 !important;'>© 2026 Trio Borma</small>", unsafe_allow_html=True)

# --- 8. LOGIC ---
if analyze_btn:
    if model is None:
        st.error("❌ Model not found.")
    else:
        with st.spinner("🛰️ Processing Satellite Data..."):
            try:
                temp_dir = Path("data/gee_temp")
                temp_dir.mkdir(parents=True, exist_ok=True)
                s1_path, dem_path, date, rain_tot, rain_avg, days = get_gee_data(lat_input, lon_input, str(temp_dir))
                
                tensor, h, w = preprocess_anti_crash(s1_path, dem_path)
                with torch.no_grad():
                    out = model(tensor.to(DEVICE))
                    prob = torch.sigmoid(out).squeeze().cpu().numpy()[:h, :w]
                
                base_sat = float(np.mean(prob) * 100)
                
                change_pct = 0.0
                logic_mode = "STABLE"

                if rain_avg < 5.0:
                    change_pct = -(1.5 * days)
                    logic_mode = "DRYING"
                else:
                    eff_days = min(days, 3)
                    calc_inc = (rain_avg * eff_days) * 0.3
                    change_pct = min(calc_inc, 25.0)
                    logic_mode = "WETTING"
                
                final_sat = max(0.0, min(100.0, base_sat + change_pct))
                
                status_text = "SAFE"
                if final_sat > 80:
                    status_text = "DANGER" if rain_avg > 15 else "WARNING"
                elif final_sat > 60:
                    status_text = "ALERT"

                st.session_state['analysis_result'] = {
                    "date": date, "base_sat": base_sat, "rain_avg": rain_avg,
                    "final_sat": final_sat, "change_pct": change_pct,
                    "status_text": status_text, "logic_mode": logic_mode,
                    "heatmap": prob, "days": days
                }
            except Exception as e:
                st.error(f"Error: {e}")

# --- 9. MAP DISPLAY ---

m = folium.Map(location=[lat_input, lon_input], zoom_start=14, tiles="CartoDB positron")
folium.Marker([lat_input, lon_input], popup="Target", icon=folium.Icon(color="black", icon="crosshairs", prefix='fa')).add_to(m)

# FIXED HEIGHT: 700px
st_data = st_folium(m, height=700, use_container_width=True, returned_objects=["last_clicked"])

if st_data['last_clicked']:
    lat_new = st_data['last_clicked']['lat']
    lon_new = st_data['last_clicked']['lng']
    if abs(lat_new - st.session_state['lat_default']) > 0.0001:
        st.session_state['lat_default'] = lat_new
        st.session_state['lon_default'] = lon_new
        st.rerun()

# --- 10. RESULTS ---
res = st.session_state['analysis_result']

if res is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader(f"📊 Audit Report: {res['status_text']}")
    st.caption(f"Data Date: {res['date']}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("AI Soil Scan", f"{res['base_sat']:.1f}%", "Satellite Base")
    c2.metric("Rainfall (Avg)", f"{res['rain_avg']:.1f} mm", "Daily Avg")
    
    delta_val = f"{res['change_pct']:+.1f}% Impact"
    col_delta = "inverse" if res['final_sat'] > 60 else "normal"
    c3.metric("FINAL SATURATION", f"{res['final_sat']:.1f}%", delta_val, delta_color=col_delta)

    st.markdown("---")
    
    c_viz, c_info = st.columns([1, 2])
    
    with c_viz:
        st.markdown("**Saturation Heatmap**")
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(res['heatmap'], cmap='jet', vmin=0, vmax=1)
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=False)
        
    with c_info:
        if "DANGER" in res['status_text']:
            st.error(f"⚠️ **CRITICAL:** High flood risk detected. Soil ({res['final_sat']:.1f}%) is saturated + heavy rain.")
        elif "ALERT" in res['status_text']:
            st.warning(f"⚠️ **WARNING:** Soil is becoming saturated.")
        else:
            st.success(f"✅ **SAFE:** Soil hydrology is stable.")
            
        # --- LOGS AREA (WARNA DIGANTI JADI PUTIH) ---
        log_text = f"""[SYSTEM LOGS]
> Target     : {lat_input:.4f}, {lon_input:.4f}
> Image Age  : {res['days']} days ago
> Logic Mode : {res['logic_mode']}
> Base Sat   : {res['base_sat']:.2f}%"""

        st.markdown(f"""
        <div style="
            background-color: #1e293b; 
            color: #ffffff; /* DIUBAH MENJADI PUTIH */
            padding: 15px; 
            border-radius: 8px; 
            font-family: 'Courier New', monospace; 
            font-size: 13px; 
            line-height: 1.6;
            border: 1px solid #334155;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <pre style="margin: 0; white-space: pre-wrap;">{log_text}</pre>
        </div>
        """, unsafe_allow_html=True)