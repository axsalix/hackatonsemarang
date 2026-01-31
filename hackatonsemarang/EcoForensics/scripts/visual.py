import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
import segmentation_models_pytorch as smp
from pathlib import Path  # <--- Library modern untuk path
from model_utils_v2 import EcoForensicsDatasetV2

# --- KONFIGURASI DINAMIS & PORTABLE ---
# 1. Deteksi lokasi file visual.py ini, lalu mundur ke folder Root 'EcoForensics'
# FILE: EcoForensics/scripts/visual.py
# PARENT: EcoForensics/scripts
# PARENT.PARENT: EcoForensics (Root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Definisikan path relatif dari Root
# Kita bungkus pakai str() agar kompatibel penuh dengan os.path di file utils
S1_DIR = str(PROJECT_ROOT / "data" / "raw" / "s1")
LABEL_DIR = str(PROJECT_ROOT / "data" / "raw" / "label")
DEM_DIR = str(PROJECT_ROOT / "data" / "raw" / "dem")
MODEL_PATH = str(PROJECT_ROOT / "models" / "sar_hydra_v2.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_prediction(num_samples=5):
    print(f"📂 Project Root: {PROJECT_ROOT}")
    print(f"🔍 Memvisualisasikan {num_samples} sampel acak...")
    
    # 1. Load Dataset
    # Path sudah otomatis benar, tidak peduli run dari mana
    if not os.path.exists(S1_DIR):
        print(f"❌ Error: Folder data tidak ditemukan di: {S1_DIR}")
        print("   Pastikan struktur folder 'data' ada di dalam 'EcoForensics'")
        return

    dataset = EcoForensicsDatasetV2(S1_DIR, LABEL_DIR, DEM_DIR)
    
    if len(dataset) == 0:
        print("❌ Dataset kosong atau semua file terfilter!")
        return

    # 2. Load Model Arsitektur
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None, 
        in_channels=4, 
        classes=1,
        activation=None
    ).to(DEVICE)

    # 3. Load Weights
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("✅ Model V2 berhasil di-load.")
    else:
        print(f"❌ File model tidak ditemukan di: {MODEL_PATH}")
        return

    model.eval()

    # 4. Loop Visualisasi
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        try:
            image_tensor, mask_tensor = dataset[idx]
            image_input = image_tensor.unsqueeze(0).to(DEVICE)

            # Prediksi
            with torch.no_grad():
                output = model(image_input)
                pred_mask = (torch.sigmoid(output) > 0.5).float().squeeze().cpu().numpy()

            # Ambil Data Asli
            vv_channel = image_tensor[0].numpy()
            dem_channel = image_tensor[2].numpy()
            ground_truth = mask_tensor.squeeze().numpy()

            # --- PLOTTING ---
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            
            # Gambar 1: Radar Asli
            ax[0].imshow(vv_channel, cmap='gray')
            ax[0].set_title(f"Input: Radar (Sample #{idx})")
            ax[0].axis('off')

            # Gambar 2: DEM
            ax[1].imshow(dem_channel, cmap='terrain')
            ax[1].set_title("Input: Topografi (DEM)")
            ax[1].axis('off')

            # Gambar 3: Kunci Jawaban
            ax[2].imshow(ground_truth, cmap='Blues', interpolation='nearest')
            ax[2].set_title("Label Asli")
            ax[2].axis('off')

            # Gambar 4: Prediksi AI
            ax[3].imshow(vv_channel, cmap='gray')
            masked_pred = np.ma.masked_where(pred_mask == 0, pred_mask)
            ax[3].imshow(masked_pred, cmap='autumn', alpha=0.6) 
            ax[3].set_title("Prediksi AI V2")
            ax[3].axis('off')

            plt.tight_layout()
            
            # Simpan output di folder root agar mudah dicari
            output_filename = PROJECT_ROOT / f"hasil_visualisasi_index_{idx}.png"
            plt.savefig(str(output_filename))
            print(f"📸 Gambar disimpan: {output_filename}")
            plt.close(fig) # Tutup plot agar hemat memori
            
        except Exception as e:
            print(f"⚠️ Error visualisasi index {idx}: {e}")

if __name__ == "__main__":
    visualize_prediction()