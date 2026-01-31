import os
import ee
import rasterio
import requests
from rasterio.warp import transform_bounds

# 1. Inisialisasi GEE (Verbose)
print("--- 🔍 DIAGNOSTIK GEE DEM ---")
try:
    ee.Initialize(project='hackaton-485815')
    print("✅ GEE Initialize Berhasil")
except Exception as e:
    print(f"❌ GEE Initialize Gagal: {e}")
    print("Coba jalankan 'earthengine authenticate' di terminal.")
    exit()

# Folder
S1_DIR = "data/raw/s1"
DEM_COLLECTION = "USGS/SRTMGL1_003"

# Ambil 1 file saja untuk tes
s1_files = sorted([f for f in os.listdir(S1_DIR) if f.endswith('.tif')])
if not s1_files:
    print("❌ Tidak ada file TIF di folder s1!")
    exit()

target_file = s1_files[0] # Ambil file pertama
s1_path = os.path.join(S1_DIR, target_file)
print(f"📄 Memeriksa file: {target_file}")

try:
    with rasterio.open(s1_path) as src:
        print(f"   - CRS Asli: {src.crs}")
        print(f"   - Bounds Asli: {src.bounds}")
        
        # Transformasi Koordinat
        left, bottom, right, top = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
        print(f"   - Bounds Lat/Lon (WGS84): {left}, {bottom}, {right}, {top}")
        
        region = ee.Geometry.Rectangle([left, bottom, right, top])
        
        # Cek apakah DEM tersedia di lokasi ini
        print("   - Menghubungi Server Google Earth Engine...")
        dem = ee.Image(DEM_COLLECTION).clip(region)
        
        # Coba generate URL
        url = dem.getDownloadURL({
            'scale': 10,
            'crs': 'EPSG:4326',
            'region': region,
            'format': 'GEO_TIFF',
            'dimensions': f"{src.width}x{src.height}"
        })
        
        print(f"✅ URL Berhasil Digenerate!")
        print(f"   Link: {url}")
        
        # Coba Download
        print("   - Mencoba mengunduh data...")
        r = requests.get(url, stream=True, timeout=30)
        
        if r.status_code == 200:
            print("🎉 SUKSES! Server merespons dengan data.")
            print("KESIMPULAN: Kode berfungsi untuk satu file. Masalah sebelumnya mungkin koneksi/timeout.")
        else:
            print(f"❌ GAGAL DOWNLOAD. Status Code: {r.status_code}")
            print(f"Pesan Error: {r.text}")

except Exception as e:
    print(f"\n❌ ERROR FATAL SAAT PROSES: {e}")
    # Tips perbaikan berdasarkan error umum
    if "Coordinate" in str(e):
        print("👉 Masalah Koordinat. Coba cek apakah file Sentinel-1 memiliki GeoReference yang benar.")
    elif "Pixel grid" in str(e):
        print("👉 Masalah Dimensi Pixel. GEE tidak bisa me-resize sesuai permintaan.")
    elif "quota" in str(e).lower():
        print("👉 Kuota GEE Habis (Quota Exceeded).")