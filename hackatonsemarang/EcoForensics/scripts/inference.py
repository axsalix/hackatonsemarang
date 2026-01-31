import os
import torch
import rasterio
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path
from tqdm import tqdm

# --- KONFIGURASI ---
# Deteksi folder root otomatis (agar tidak error path)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder Input (Pastikan Anda menaruh data tes di sini)
INPUT_S1_DIR = PROJECT_ROOT / "data" / "inference" / "s1"
INPUT_DEM_DIR = PROJECT_ROOT / "data" / "inference" / "dem"

# Folder Output
OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
MODEL_PATH = PROJECT_ROOT / "models" / "sar_hydra_v2.pth"

# Pastikan folder ada
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Memuat arsitektur model yang SAMA PERSIS dengan saat training."""
    print(f"⏳ Memuat model dari: {MODEL_PATH}")
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=4,  # Sesuai training (VV, VH, DEM, Moisture)
        classes=1,
        activation=None
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model tidak ditemukan di {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model berhasil dimuat!")
    return model

def preprocess_input(s1_path, dem_path):
    """
    Melakukan preprocessing yang SAMA PERSIS dengan model_utils_v2.py.
    PENTING: Rumus matematika tidak boleh beda 1 angka pun.
    """
    # 1. Load S1 (Band 1 & 2)
    with rasterio.open(s1_path) as src:
        s1 = src.read([1, 2]).astype(np.float32)
        profile = src.profile # Simpan metadata koordinat (PENTING untuk GIS)
        
        # Handle NaN & Normalisasi S1
        s1 = np.nan_to_num(s1, nan=0.0)
        s1 = np.clip(s1, -30, 0)
        s1 = (s1 + 30) / 30.0

    # 2. Load DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        # Handle NaN & Normalisasi DEM
        dem = np.nan_to_num(dem, nan=0.0)
        dem = np.clip(dem, 0, 1000) / 1000.0

    # 3. Buat Channel Soil Moisture Proxy (Sesuai training)
    soil_moisture = s1[0]

    # 4. Stacking menjadi 4 Channel
    # Shape: (4, H, W)
    input_tensor = np.stack([s1[0], s1[1], dem, soil_moisture], axis=0)
    input_tensor = np.nan_to_num(input_tensor, nan=0.0)
    
    # Ubah ke Torch Tensor (tambah batch dimension) -> (1, 4, H, W)
    tensor = torch.from_numpy(input_tensor).unsqueeze(0).float()
    
    return tensor, profile

def save_prediction(pred_mask, profile, filename):
    """Menyimpan hasil prediksi sebagai GeoTIFF (punya koordinat)."""
    # Update profile agar sesuai format output (1 channel, float32)
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw'
    )

    output_path = OUTPUT_DIR / filename
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(pred_mask, 1)
    
    return output_path

def main():
    print("🚀 Memulai Eco-Forensics Inference Engine...")
    
    # 1. Load Model
    try:
        model = load_model()
    except Exception as e:
        print(e)
        return

    # 2. Cari File Input
    if not os.path.exists(INPUT_S1_DIR):
        print(f"❌ Folder input tidak ditemukan: {INPUT_S1_DIR}")
        print("   Buat folder tersebut dan masukkan file .tif S1 dan DEM.")
        return

    s1_files = sorted([f for f in os.listdir(INPUT_S1_DIR) if f.endswith('.tif')])
    print(f"📂 Ditemukan {len(s1_files)} file citra satelit untuk diproses.")

    # 3. Loop Inference
    for f_name in tqdm(s1_files, desc="Processing"):
        s1_path = INPUT_S1_DIR / f_name
        
        # Cari pasangan DEM-nya (Asumsi penamaan file konsisten seperti training)
        # Jika nama file S1: "S1Hand_2025.tif", maka DEM: "DEM_2025.tif"
        dem_name = f_name.replace("S1Hand", "DEM") 
        dem_path = INPUT_DEM_DIR / dem_name

        if not dem_path.exists():
            print(f"⚠️ Skip {f_name}: File DEM tidak ditemukan ({dem_name})")
            continue

        try:
            # A. Preprocessing
            input_tensor, profile = preprocess_input(s1_path, dem_path)
            input_tensor = input_tensor.to(DEVICE)

            # B. Prediksi AI
            with torch.no_grad():
                output = model(input_tensor)
                # Output sigmoid: 0.0 (Kering) s/d 1.0 (Banjir Total)
                # Kita simpan probabilitasnya untuk Heatmap Dashboard
                pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()

            # C. Simpan Hasil (GeoTIFF)
            output_filename = f_name.replace("S1Hand", "Pred_FloodRisk")
            saved_path = save_prediction(pred_prob, profile, output_filename)
            
            # Opsional: Simpan juga versi Binary (Masking murni 0 dan 1)
            # Threshold 0.5 sesuai proposal
            binary_mask = (pred_prob > 0.5).astype(np.float32)
            save_prediction(binary_mask, profile, f"Binary_{output_filename}")

        except Exception as e:
            print(f"❌ Error pada file {f_name}: {e}")

    print(f"\n✅ Selesai! Hasil prediksi tersimpan di: {OUTPUT_DIR}")
    print("   File 'Pred_FloodRisk_...' berisi probabilitas (untuk Heatmap).")
    print("   File 'Binary_...' berisi mask 0/1 (untuk deteksi area tegas).")

if __name__ == "__main__":
    main()