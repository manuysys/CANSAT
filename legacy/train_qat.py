"""Quantization-Aware Training: fine-tune simulando quantización INT8."""

# ── legacy/: script fuera del pipeline activo. Para correrlo desde la raíz:
#     python legacy/train_qat.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import torch
from torch import nn
import torch.quantization as quant
from train import LoveDADataset, confusion_of, metrics_from_conf, IGNORE_INDEX
from train_cbam import DeepLabV3PlusCBAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando {device}")

# Cargar modelo base
model = DeepLabV3PlusCBAM(5).to(device)
model.load_state_dict(torch.load("outputs/best_terrain_cbam.pth", map_location=device))

# Preparar para QAT
model.train()
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
quant.prepare_qat(model, inplace=True)
print("[OK] Modelo preparado para QAT")

# Dataset
ds = LoveDADataset("dataset/loveda_remapped", "Train", 320, augment=True)
dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

# Loss y optimizer
weight = torch.tensor([0.347, 1.137, 1.966, 1.199, 0.351]).to(device)
crit = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)
opt = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

# Entrenar 10 epochs
print("Entrenando con QAT (10 epochs)...")
for ep in range(10):
    total_loss = 0
    for i, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        total_loss += loss.item()

        if i % 50 == 0:
            print(f"  epoch {ep+1} batch {i}/{len(dl)} loss {loss.item():.3f}", flush=True)

    avg_loss = total_loss / len(dl)
    print(f"epoch {ep+1}: loss {avg_loss:.3f}")

# Convertir a quantized
model.eval()
quant.convert(model, inplace=True)
torch.save(model.state_dict(), "outputs/best_terrain_cbam_qat.pth")
print("[OK] Modelo QAT guardado")

# Evaluar en Val
val_ds = LoveDADataset("dataset/loveda_remapped", "Val", 320, augment=False)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=8, num_workers=0)
conf = confusion_of(model, val_dl, device)
miou, iou, f1 = metrics_from_conf(conf)
print(f"\nQAT FP32: mIoU {miou:.4f} | " + " ".join(f"{v:.3f}" for v in iou))
print("FP32 original: 0.5317")
