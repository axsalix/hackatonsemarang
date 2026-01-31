import os
import ee
import rasterio
import requests
import time
from rasterio.warp import transform_bounds
from tqdm import tqdm

# 1. Inisialisasi GEE
try:
    ee.Initialize(project='hackaton-485815')
except:
    ee.Authenticate()
    ee.Initialize()

# Konfigurasi
RAW_DIR = "data/raw"
S1_DIR = os.path.join(RAW_DIR, "s1")
DEM_DIR = os.path.join(RAW_DIR, "dem")
os.makedirs(DEM_DIR, exist_ok=True)

# Dataset Topografi NASA SRTM
DEM_COLLECTION = "USGS/SRTMGL1_003"

def get_dem_url(bounds_latlon, width, height, retries=3):
    region = ee.Geometry.Rectangle(bounds_latlon)
    dem = ee.Image(DEM_COLLECTION).clip(region)
    
    for i in range(retries):
        try:
            # PERBAIKAN DI SINI: Hapus 'scale': 10
            # Kita hanya minta dimensions agar pixel match dengan Sentinel-1
            url = dem.getDownloadURL({
                'crs': 'EPSG:4326', 
                'region': region,
                'format': 'GEO_TIFF',
                'dimensions': f"{width}x{height}" 
            })
            return url
        except Exception as e:
            if i < retries - 1:
                time.sleep(2)
            else:
                return None
    return None

def download_file(url, save_path):
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except:
        pass
    return False

print("--- 🛰️ FINAL V2: DOWNLOAD REAL DEM (FIXED PARAMETERS) ---")

s1_files = sorted([f for f in os.listdir(S1_DIR) if f.endswith('.tif')])
success_count = 0
fail_count = 0

# Loop download
for s1_file in tqdm(s1_files, desc="Downloading DEMs"):
    s1_path = os.path.join(S1_DIR, s1_file)
    dem_path = os.path.join(DEM_DIR, s1_file.replace("S1Hand", "DEM"))
    
    # Skip jika sudah ada & valid
    if os.path.exists(dem_path):
        try:
            with rasterio.open(dem_path) as src:
                src.read(1)
            success_count += 1
            continue 
        except:
            pass 

    try:
        with rasterio.open(s1_path) as src:
            width = src.width
            height = src.height
            
            # Transformasi Koordinat (Jaga-jaga jika bukan EPSG:4326)
            left, bottom, right, top = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            bounds_latlon = [left, bottom, right, top]
        
        # Minta URL
        url = get_dem_url(bounds_latlon, width, height)
        
        if url:
            if download_file(url, dem_path):
                success_count += 1
            else:
                fail_count += 1
        else:
            print(f"URL Gagal untuk {s1_file}")
            fail_count += 1
            
        time.sleep(0.5) 

    except Exception as e:
        print(f"Error {s1_file}: {e}")
        fail_count += 1

print(f"\n✅ SELESAI! Berhasil: {success_count}, Gagal: {fail_count}")