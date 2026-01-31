import ee
import requests
import os
from datetime import datetime

# Inisialisasi GEE
try:
    ee.Initialize(project='trioborma1')
except Exception:
    ee.Authenticate()
    ee.Initialize()

def get_realtime_rain_history(lat, lon, s1_date):
    """
    Mengambil data hujan real-time (Forecast/Past Days) untuk mengatasi masalah 0.00mm.
    """
    try:
        s1_dt = datetime.strptime(s1_date, '%Y-%m-%d')
        today_dt = datetime.now()
        delta_days = (today_dt - s1_dt).days
        
        if delta_days < 1: delta_days = 1
        if delta_days > 90: delta_days = 90
        
        print(f"🌦️ Menarik data hujan {delta_days} hari terakhir...")

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, 
            "longitude": lon,
            "past_days": delta_days,
            "forecast_days": 1,
            "daily": "precipitation_sum",
            "timezone": "GMT"
        }
        
        res = requests.get(url, params=params, timeout=10).json()
        rain_list = res.get('daily', {}).get('precipitation_sum', [])
        
        clean_rain = [x if x is not None else 0.0 for x in rain_list]
        total_rain = sum(clean_rain)
        avg_rain = total_rain / len(clean_rain) if len(clean_rain) > 0 else 0.0
        
        print(f"✅ Data Hujan: Total {total_rain:.2f} mm")
        return total_rain, avg_rain, len(clean_rain)

    except Exception as e:
        print(f"❌ Error API Hujan: {e}")
        return 0.0, 0.0, 1

def get_gee_data(lat, lon, output_dir, buffer_km=5):
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_km * 1000).bounds()

    # 1. Sentinel-1
    s1_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(region)
              .filter(ee.Filter.eq('instrumentMode', 'IW'))
              .sort('system:time_start', False))
    
    img_s1 = s1_col.first()
    s1_date = img_s1.date().format('YYYY-MM-dd').getInfo()

    # --- PERBAIKAN DI SINI (Fungsi Download Fleksibel) ---
    def download(img, name, band_name):
        # Kita select band sesuai parameter, bukan hardcode 'VV'
        url = img.select([band_name]).getDownloadURL({
            'scale': 30, 
            'region': region, 
            'format': 'GEO_TIFF'
        })
        path = os.path.join(output_dir, name)
        with open(path, 'wb') as f: f.write(requests.get(url).content)
        return path

    dem_img = ee.Image("NASA/NASADEM_HGT/001")
    
    print(f"🛰️ Mengunduh data S1 tanggal {s1_date}...")
    
    # Panggil fungsi download dengan band yang benar
    s1_path = download(img_s1, f"S1_{lat}_{lon}.tif", 'VV')         # Sentinel pakai VV
    dem_path = download(dem_img, f"DEM_{lat}_{lon}.tif", 'elevation') # DEM pakai elevation

    # 3. Ambil Hujan
    total, avg, days = get_realtime_rain_history(lat, lon, s1_date)

    return s1_path, dem_path, s1_date, total, avg, days