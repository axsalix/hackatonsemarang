import os
import pandas as pd
import requests
from tqdm import tqdm

# 1. Folder penyimpanan di laptop kamu
base_raw = "data/raw"
s1_dir = os.path.join(base_raw, "s1")
label_dir = os.path.join(base_raw, "label")

os.makedirs(s1_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

def download_tif(url, save_path):
    if os.path.exists(save_path):
        return
    try:
        # Timeout ditambah agar lebih stabil
        r = requests.get(url, stream=True, timeout=20)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            # Jika masih 404, kita akan tahu alamat mana yang salah
            print(f"\n❌ Gagal ({r.status_code}): {url}")
    except Exception as e:
        print(f"\n❌ Error Koneksi: {e}")

# 2. Baca daftar file dari CSV Metadata
csv_path = os.path.join(base_raw, "flood_train_data.csv")
# Kita gunakan names=['s1', 'label'] agar pemanggilannya konsisten
df = pd.read_csv(csv_path, header=None, names=['s1', 'label'])

# 3. Alamat Dasar (Base URL) Server Google Cloud yang SUDAH DIPERBAIKI
base_url = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled"

print(f"--- Mengunduh 10 Sampel Gambar Sentinel-1 & Label ---")

# Ambil 250 data agar model lebih generalis
samples = df.head(250)

for idx, row in tqdm(samples.iterrows(), total=len(samples)):
    s1_name = str(row['s1']).strip()
    label_name = str(row['label']).strip()
    
    # PERBAIKAN: Folder di server adalah S1Hand dan LabelHand
    s1_url = f"{base_url}/S1Hand/{s1_name}"
    download_tif(s1_url, os.path.join(s1_dir, s1_name))
    
    label_url = f"{base_url}/LabelHand/{label_name}"
    download_tif(label_url, os.path.join(label_dir, label_name))

print(f"\n🚀 AKHIRNYA BERHASIL! Silakan cek folder:")
print(f"📁 {s1_dir}")
print(f"📁 {label_dir}")