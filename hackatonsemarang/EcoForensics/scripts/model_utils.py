import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset

class FloodDataset(Dataset):
    def __init__(self, s1_dir, label_dir):
        self.s1_dir = s1_dir
        self.label_dir = label_dir
        s1_files = set(os.listdir(s1_dir))
        label_files = set(os.listdir(label_dir))
        
        self.valid_files = []
        for f in sorted(list(s1_files)):
            label_name = f.replace("S1Hand.tif", "LabelHand.tif")
            if label_name in label_files:
                self.valid_files.append((f, label_name))

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        s1_name, label_name = self.valid_files[idx]
        s1_path = os.path.join(self.s1_dir, s1_name)
        label_path = os.path.join(self.label_dir, label_name)

        try:
            with rasterio.open(s1_path) as src:
                image = src.read([1, 2]).astype(np.float32)
                
                # PROTEKSI TOTAL: Ubah NaN atau Inf menjadi 0
                image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Normalisasi yang sangat stabil
                image = np.clip(image, -35, 0) # Rentang radar yang lebih aman
                image = (image + 35) / 35.0

            with rasterio.open(label_path) as src:
                mask = src.read(1).astype(np.float32)
                mask = np.nan_to_num(mask, nan=0.0)
                mask = np.where(mask > 0, 1, 0).astype(np.float32)
            
            return torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0)

        except Exception:
            return self.__getitem__((idx + 1) % len(self.valid_files))

print("✅ Model Utils: Dataset Loader Berhasil Dibuat!")