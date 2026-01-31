import os
import requests
from tqdm import tqdm

def download_file(url, folder):
    local_filename = os.path.join(folder, url.split('/')[-1])
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_filename, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=url.split('/')[-1]
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"✅ Berhasil: {local_filename}")
    except Exception as e:
        print(f"❌ Gagal mengunduh {url}: {e}")

# Folder penyimpanan
raw_dir = "data/raw"
os.makedirs(raw_dir, exist_ok=True)

# Link RESMI dari Google Cloud Storage Sen1Floods11
# Ini adalah daftar 'Hand Labeled' (Data kualitas tinggi yang dikurasi manusia)
base_url = "https://storage.googleapis.com/sen1floods11/v1.1/splits/flood_handlabeled/"
files = [
    "flood_train_data.csv",
    "flood_valid_data.csv",
    "flood_test_data.csv"
]

print("--- Memulai Pengunduhan Metadata dari Google Storage ---")
for f_name in files:
    download_file(base_url + f_name, raw_dir)

print("\n--- Langkah 1.4 Selesai ---")