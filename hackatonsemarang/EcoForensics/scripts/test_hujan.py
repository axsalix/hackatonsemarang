import ee
import requests
import pandas as pd
from datetime import datetime, timedelta

# 1. Inisialisasi GEE
try:
    ee.Initialize(project = 'trioborma1')
except Exception:
    ee.Authenticate()
    ee.Initialize()

def fetch_weather_data(lat, lon, start_date, end_date):
    """Menarik data hujan harian dari Open-Meteo"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "GMT"
    }
    res = requests.get(url, params=params)
    return res.json().get('daily', {}).get('precipitation_sum', [])

def data_acquisition_audit(lat, lon):
    print(f"\n{'='*60}")
    print(f"🔍 ECO-FORENSICS DATA AUDIT: {lat}, {lon}")
    print(f"{'='*60}")

    # --- A. SATELIT SENTINEL-1 (GEE) ---
    print("\n🛰️ [DATA SATELIT: Sentinel-1]")
    point = ee.Geometry.Point([lon, lat])
    s1_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(point)
              .filter(ee.Filter.eq('instrumentMode', 'IW'))
              .sort('system:time_start', False))

    # Ambil 2 image terbaru
    s1_list = s1_col.toList(2).getInfo()
    
    if len(s1_list) < 2:
        print("❌ Data Sentinel-1 tidak mencukupi di lokasi ini.")
        return

    # Info Cycle 2 (Terbaru)
    c2_date = datetime.fromtimestamp(s1_list[0]['properties']['system:time_start']/1000).strftime('%Y-%m-%d')
    c2_id = s1_list[0]['id']
    
    # Info Cycle 1 (Sebelumnya)
    c1_date = datetime.fromtimestamp(s1_list[1]['properties']['system:time_start']/1000).strftime('%Y-%m-%d')
    c1_id = s1_list[1]['id']

    print(f"✅ Cycle 2 (Terbaru) : {c2_date} [ID: {c2_id}]")
    print(f"✅ Cycle 1 (Lama)    : {c1_date} [ID: {c1_id}]")

    # --- B. TOPOGRAFI (DEM) ---
    print("\n🏔️ [DATA TOPOGRAFI: NASADEM]")
    dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
    elev = dem.reduceRegion(ee.Reducer.mean(), point, 30).getInfo().get('elevation')
    print(f"✅ Ketinggian Rata-rata: {elev:.2f} meter di atas permukaan laut")

    # --- C. CURAH HUJAN (API JSON) ---
    print("\n🌦️ [DATA HUJAN: Open-Meteo Archive]")
    
    # 1. Hujan di antara Cycle (untuk hitung K-Ratio)
    rain_inter = fetch_weather_data(lat, lon, c1_date, c2_date)
    total_rain_inter = sum(rain_inter)
    
    # 2. Hujan sejak Cycle terakhir s/d Kemarin (untuk Prediksi)
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    rain_recent = fetch_weather_data(lat, lon, c2_date, yesterday)
    total_rain_recent = sum(rain_recent)

    print(f"✅ Hujan di antara Cycle      : {total_rain_inter:.2f} mm ({len(rain_inter)} hari)")
    print(f"✅ Hujan s/d Kemarin ({yesterday}) : {total_rain_recent:.2f} mm ({len(rain_recent)} hari)")

    # --- D. KESIMPULAN AUDIT ---
    print(f"\n{'='*60}")
    print("📊 KESIMPULAN INTEGRASI DATA:")
    print(f"1. Model AI akan membandingkan gambar radar {c1_date} vs {c2_date}.")
    print(f"2. Faktor pengali (K-Ratio) dihitung dari {total_rain_inter:.2f} mm hujan.")
    print(f"3. Estimasi banjir hari ini dipicu oleh akumulasi {total_rain_recent:.2f} mm hujan.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Test Koordinat (Contoh: Area rawan banjir di Semarang)
    data_acquisition_audit(-6.96, 110.41)