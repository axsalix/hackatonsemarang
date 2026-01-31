import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from model_utils import FloodDataset
import os

# 1. Konfigurasi
EPOCHS = 100  # Tambah epoch agar lebih matang
BATCH_SIZE = 4 
LEARNING_RATE = 0.0001 # Lebih kecil agar belajar lebih teliti

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load & Split Data
full_dataset = FloodDataset("data/raw/s1", "data/raw/label")
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# 3. Model & Optimizer
model = smp.Unet(encoder_name="resnet34", in_channels=2, classes=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

# 4. Fungsi Hitung IoU (Senjata buat jawab juri)
def get_iou(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    if union == 0: return 1.0
    return (intersection / union).item()

# 5. Training Loop
print(f"🚀 Memulai Training dengan {len(full_dataset)} gambar...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validasi (Ujian AI)
    model.eval()
    val_iou = 0
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            val_iou += get_iou(output, mask)
    
    avg_loss = train_loss/len(train_loader)
    avg_iou = val_iou/len(val_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Val IoU: {avg_iou:.4f}")

torch.save(model.state_dict(), "models/sar_hydra_final.pth")
print("\n✅ MODEL FINAL SIAP DIBAWA KE SEMARANG!")