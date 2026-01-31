import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset

class EcoForensicsDatasetV2(Dataset):
    def __init__(self, s1_dir, label_dir, dem_dir):
        self.s1_dir = s1_dir
        self.label_dir = label_dir
        self.dem_dir = dem_dir
        
        self.files = []
        s1_all = sorted(os.listdir(s1_dir))
        
        # --- FILTER DATA DI AWAL (Biar Training Cepat) ---
        print("🧹 Memeriksa kualitas data...")
        valid_count = 0
        
        for f in s1_all:
            if not f.endswith('.tif'): continue
            
            s1_path = os.path.join(s1_dir, f)
            label_name = f.replace("S1Hand", "LabelHand")
            dem_name = f.replace("S1Hand", "DEM")
            
            # Cek 1: Apakah file lengkap?
            if not os.path.exists(os.path.join(dem_dir, dem_name)):
                continue

            # Cek 2: APAKAH GAMBARNYA KOSONG/HITAM? (Penting!)
            # Kita intip sedikit isinya tanpa load semua
            try:
                with rasterio.open(s1_path) as src:
                    # Baca versi kecil (thumbnail) biar cepat
                    overview = src.read(1, out_shape=(1, int(src.height // 10), int(src.width // 10)))
                    
                    # Hitung persentase piksel bernilai 0 (Data Kosong)
                    zero_pixels = np.count_nonzero(overview == 0)
                    total_pixels = overview.size
                    empty_ratio = zero_pixels / total_pixels
                    
                    # Jika lebih dari 30% gambar adalah hitam kosong -> BUANG
                    if empty_ratio > 0.3: 
                        continue
                        
                self.files.append((f, label_name, dem_name))
                valid_count += 1
                
            except:
                continue
        
        print(f"✅ Dataset Bersih: {valid_count} gambar (Sisanya dibuang karena kosong/rusak)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            s1_name, label_name, dem_name = self.files[idx]
            
            # 1. LOAD S1
            with rasterio.open(os.path.join(self.s1_dir, s1_name)) as src:
                s1 = src.read([1, 2]).astype(np.float32)
                # Handle NaN & Noise
                s1 = np.nan_to_num(s1, nan=0.0)
                # Normalisasi
                s1 = np.clip(s1, -30, 0)
                s1 = (s1 + 30) / 30.0 

            # 2. LOAD DEM
            with rasterio.open(os.path.join(self.dem_dir, dem_name)) as src:
                dem = src.read(1).astype(np.float32)
                dem = np.nan_to_num(dem, nan=0.0)
                dem = np.clip(dem, 0, 1000) / 1000.0 

            # 3. SOIL MOISTURE PROXY
            soil_moisture = s1[0] 
            
            # 4. STACK
            input_tensor = np.stack([s1[0], s1[1], dem, soil_moisture], axis=0)
            
            # Pengaman Tensor (Biar ga error loss NaN)
            input_tensor = np.nan_to_num(input_tensor, nan=0.0)

            # 5. LOAD LABEL
            with rasterio.open(os.path.join(self.label_dir, label_name)) as src:
                mask = src.read(1).astype(np.float32)
                mask = (mask > 0).astype(np.float32)

            return torch.from_numpy(input_tensor), torch.from_numpy(mask).unsqueeze(0)

        except Exception as e:
            # Recursive skip jika error
            new_idx = (idx + 1) % len(self.files)
            return self.__getitem__(new_idx)