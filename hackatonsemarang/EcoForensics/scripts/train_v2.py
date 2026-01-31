import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from model_utils_v2 import EcoForensicsDatasetV2
import os
from tqdm import tqdm

# --- FUNGSI UTAMA (Dibungkus agar aman di Windows) ---
def main():
    # --- CONFIG ---
    EPOCHS = 40          
    BATCH_SIZE = 8
    # TURUNKAN LEARNING RATE (Biar loss ga naik)
    LEARNING_RATE = 1e-5  # Sebelumnya 1e-4, kita perlambat 10x biar lebih teliti
    ENCODER = "efficientnet-b4"
    ENCODER_WEIGHTS = "imagenet"

    # Folder
    S1_DIR = "data/raw/s1"
    LABEL_DIR = "data/raw/label"
    DEM_DIR = "data/raw/dem"

    # Cek Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 DEVICE: GPU {torch.cuda.get_device_name(0)} (Akhirnya!)")
    else:
        device = torch.device("cpu")
        print("⚠️ DEVICE: CPU (Training akan lambat. Cek versi Python Anda!)")

    # Dataset V2
    full_dataset = EcoForensicsDatasetV2(S1_DIR, LABEL_DIR, DEM_DIR)
    print(f"📊 Jumlah Data dengan DEM: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("❌ ERROR: Dataset kosong! Pastikan script download DEM sudah selesai.")
        return

    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # PENTING UNTUK WINDOWS: num_workers=0 jika masih error, tapi coba 4 dulu
    # pin_memory=True membantu transfer data ke GPU
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    # MODEL V2
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4", 
        encoder_weights="imagenet", 
        in_channels=4,     
        classes=1, 
        activation=None
    ).to(device)

    # SHOCK THERAPY: POS_WEIGHT
    # Ini memaksa AI: "Hei, 1 piksel banjir itu seharga 20 piksel daratan!"
    # Jangan abaikan banjir!
    pos_weight = torch.tensor([20.0]).to(device) 
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # Training Loop
    best_iou = 0.0
    print("🚀 STARTING TRAINING V2...")

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for img, mask in loop:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, mask)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        
        # Validasi
        model.eval()
        tp, fp, fn, tn = 0, 0, 0, 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                preds = (torch.sigmoid(model(img)) > 0.5).long()
                stats = smp.metrics.get_stats(preds, mask.long(), mode='binary', threshold=0.5)
                tp += stats[0].sum()
                fp += stats[1].sum()
                fn += stats[2].sum()
                tn += stats[3].sum()

        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        print(f"   🏆 Val IoU: {iou_score:.4f}")

        if iou_score > best_iou:
            best_iou = iou_score
            torch.save(model.state_dict(), "models/sar_hydra_v2.pth")
            print("   💾 Model V2 Tersimpan!")

    print("✅ TRAINING SELESAI!")

if __name__ == '__main__':
    # Ini pagar wajib untuk Windows Multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()