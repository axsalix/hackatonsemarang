import sys
import os
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import segmentation_models_pytorch as smp

# --- 1. SETUP IMPORT (Supaya gee_loader terbaca) ---
current_file_path = Path(__file__).resolve()
current_folder = current_file_path.parent  # Folder scripts/
if str(current_folder) not in sys.path:
    sys.path.append(str(current_folder))

try:
    from gee_loader import get_gee_data
except ImportError:
    print("❌ CRITICAL: gee_loader.py tidak ditemukan!")

# --- 2. SETUP PATH ---
PROJECT_ROOT = current_folder.parent
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "gee_temp"
OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
MODEL_PATH = PROJECT_ROOT / "models" / "sar_hydra_v2.pth"
TEMPLATE_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. INIT APP ---
app = FastAPI(title="Eco-Forensics Enterprise")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/predictions", StaticFiles(directory=OUTPUT_DIR), name="predictions")

# --- 4. LOAD MODEL ---
print(f"🔄 Loading Model di {DEVICE}...")
model = smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=4, classes=1).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

# --- 5. PREPROCESSING (INI YANG DIPERBAIKI) ---
def pad_image(array, divisor=32):
    c, h, w = array.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h == 0 and pad_w == 0: return array
    return np.pad(array, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

def preprocess_gee_data(s1_path, dem_path):
    # FIX: Baca Sentinel-1 dengan penanganan Single Band
    with rasterio.open(s1_path) as src:
        bands = src.read() # Shape: (Channel, H, W)
        
        # --- PERBAIKAN ERROR INDEX DISINI ---
        if bands.shape[0] < 2:
            # Jika cuma ada 1 lapis, duplikasi lapis tsb biar jadi 2
            print("⚠️ WARNING: Single band image detected. Duplicating band...")
            s1 = np.concatenate([bands, bands], axis=0) 
        else:
            # Jika ada 2 atau lebih, ambil 2 pertama
            s1 = bands[:2]
            
        # Normalisasi
        s1 = (np.clip(s1, -30, 0) + 30) / 30.0

    # Baca DEM
    with rasterio.open(dem_path) as src:
        dem = np.clip(src.read(1), 0, 1000) / 1000.0

    # Resize DEM jika beda ukuran
    target_h, target_w = s1.shape[1], s1.shape[2]
    if dem.shape != s1.shape[1:]:
        import cv2
        dem = cv2.resize(dem, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Susun Tensor: [VV, VH, DEM, VV]
    # Karena s1 sudah pasti punya index 0 dan 1 (berkat fix diatas), ini aman
    input_stack = np.stack([s1[0], s1[1], dem, s1[0]])
    
    tensor = torch.from_numpy(pad_image(input_stack)).unsqueeze(0).float()
    return tensor, s1.shape[1:]

# --- 6. ENDPOINTS ---
@app.get("/")
async def index():
    return FileResponse(TEMPLATE_DIR / "index.html")

@app.get("/search")
async def search_proxy(q: str):
    try:
        headers = {'User-Agent': 'EcoForensicsApp/1.0'}
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': q, 'format': 'json', 'limit': 1}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        return r.json() if r.status_code == 200 else []
    except:
        return []

@app.get("/analyze")
async def analyze(lat: float, lon: float):
    try:
        # A. AMBIL DATA
        s1_path, dem_path, date, rain_tot, rain_avg, days = get_gee_data(lat, lon, str(DOWNLOAD_DIR))
        
        # B. INFERENCE (Aman dari IndexError)
        tensor, size = preprocess_gee_data(s1_path, dem_path)
        with torch.no_grad():
            out = model(tensor.to(DEVICE))
            prob = torch.sigmoid(out).squeeze().cpu().numpy()[:size[0], :size[1]]

        base_sat = float(np.mean(prob) * 100)
        
        # C. LOGIKA HUJAN (Safe Mode)
        change_pct = 0.0
        logic_mode = "STABLE"

        print(f"DEBUG: Rain {rain_avg} mm, Days {days}")

        if rain_avg < 5.0:
            change_pct = -(1.5 * days)
            logic_mode = "DRYING (Evaporasi)"
        else:
            # Max calculate 3 hari kebelakang
            eff_days = min(days, 3)
            # Rumus kalem
            calc_inc = (rain_avg * eff_days) * 0.3
            # Mentok di +25%
            change_pct = min(calc_inc, 25.0)
            logic_mode = "WETTING (Akumulasi)"
        
        final_sat = max(0.0, min(100.0, base_sat + change_pct))
        
        # D. STATUS
        status, msg = "AMAN", "Stabil."
        if final_sat > 80:
            status = "BAHAYA" if rain_avg > 15 else "SIAGA"
            msg = "Potensi banjir!" if status == "BAHAYA" else "Tanah jenuh."
        elif final_sat > 60:
            status, msg = "WASPADA", "Tanah basah."

        # Simpan Gambar
        filename = f"map_{lat}_{lon}.png"
        plt.imsave(OUTPUT_DIR / filename, prob, cmap='jet', vmin=0, vmax=1)

        return {
            "lat": lat, "lon": lon, "date": date,
            "avg_rain": float(rain_avg),
            "base_saturation": float(base_sat),
            "rain_increase": float(change_pct),
            "avg_risk": float(final_sat),
            "status": status, "message": msg, "logic": logic_mode,
            "image_url": f"/predictions/{filename}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)