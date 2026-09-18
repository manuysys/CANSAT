"""
CanSat La Base - PoC segmentación semántica 5 clases (LoveDA remapeado)
Arquitectura: DeepLabV3+ / MobileNetV2 (output stride 8)
Mismo familia de ops que el DeepLabV3Plus del model zoo oficial del IMX500.
"""
import os
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cansat.seed import device as pick_device, set_seed

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

CLASS_NAMES = ["vegetation", "building", "water", "bare_ground", "other"]
NUM_CLASSES = 5
IGNORE_INDEX = 255


# ───────────────────────── Dataset ─────────────────────────
class LoveDADataset(Dataset):
    def __init__(self, root, split, img_size, augment=False):
        self.img_size = img_size
        self.augment = augment
        self.samples = []
        for env in ["Rural", "Urban"]:
            img_dir = os.path.join(root, split, env, "images_png")
            mask_dir = os.path.join(root, split, env, "masks_png")
            if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
                continue
            for f in sorted(os.listdir(img_dir)):
                if f.lower().endswith(".png"):
                    mp = os.path.join(mask_dir, f)
                    if os.path.exists(mp):
                        self.samples.append((os.path.join(img_dir, f), mp))
        if not self.samples:
            raise FileNotFoundError(f"No hay muestras en {root}/{split}")
        print(f"[Dataset] {split}: {len(self.samples)} muestras")

        self.jitter = transforms.ColorJitter(0.15, 0.15, 0.15)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(mask_path).resize((self.img_size, self.img_size), Image.NEAREST)

        img = np.array(img)
        mask = np.array(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[::-1, :])
                mask = np.ascontiguousarray(mask[::-1, :])
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])
                img = np.ascontiguousarray(np.rot90(img, k))
                mask = np.ascontiguousarray(np.rot90(mask, k))
            img = self.jitter(Image.fromarray(img))
        else:
            img = Image.fromarray(img)

        img_t = self.norm(transforms.ToTensor()(img))
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return img_t, mask_t


# ───────────────────────── Modelo ─────────────────────────
class _ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ASPPPool(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch, rates=(6, 12, 18), out_ch=256):
        super().__init__()
        branches = [nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))]
        for r in rates:
            branches.append(_ASPPConv(in_ch, out_ch, r))
        branches.append(_ASPPPool(in_ch, out_ch))
        self.convs = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(len(branches) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
        )

    def forward(self, x):
        return self.project(torch.cat([c(x) for c in self.convs], dim=1))


class DeepLabV3PlusMobileNetV2(nn.Module):
    """DeepLabV3+ MobileNetV2, output stride 8, decoder con features de stride 4."""
    def __init__(self, num_classes):
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        # Output stride 8: dilatar los dos últimos stages con stride 2
        for idx in range(7, 18):
            for m in backbone[idx].modules():
                if isinstance(m, nn.Conv2d) and m.groups == m.in_channels and m.kernel_size == (3, 3):
                    if m.stride == (2, 2):
                        m.stride = (1, 1)
                    d = 2 if idx <= 13 else 4
                    m.dilation = (d, d)
                    m.padding = (d, d)
        self.backbone = backbone
        self.aspp = ASPP(1280, (6, 12, 18), 256)
        self.low_conv = nn.Sequential(nn.Conv2d(24, 48, 1, bias=False),
                                      nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        low = None
        for idx, block in enumerate(self.backbone):
            x = block(x)
            if idx == 3:          # features de stride 4, 24 canales
                low = x
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, self.low_conv(low)], dim=1)
        x = self.decoder(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


# ─────────────────── Pesos de clase (cache) ───────────────────
def _dataset_fingerprint(dataset_root, split="Train"):
    """
    Huella del dataset, para invalidar la caché de pesos cuando cambia.

    ⚠ ANTES: ``class_weights.npy`` se cacheaba en la RAÍZ DEL REPO (y estaba
      commiteado). Si cambiabas el remapeo de clases o el dataset, el script
      recargaba los pesos viejos **en silencio** y entrenabas con una ponderación
      que no correspondía. Ahora la huella va en el nombre del archivo, así que
      un dataset distinto produce un cache distinto y nunca se confunde.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(str(Path(dataset_root).resolve()).encode())
    h.update(split.encode())
    n = 0
    for env in ("Rural", "Urban"):
        d = Path(dataset_root) / split / env / "masks_png"
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.png")):
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
            n += 1
            if n >= 500:          # acotado: no stat-ear 5000 archivos siempre
                break
        if n >= 500:
            break
    h.update(str(n).encode())
    return h.hexdigest()[:12], n


def class_weights(dataset_root, split="Train", cache_dir="outputs", force=False):
    """Pesos inversos a la frecuencia de clases, con caché invalidada por dataset."""
    fp, n_files = _dataset_fingerprint(dataset_root, split)
    cache = Path(cache_dir) / f"class_weights_{split}_{fp}.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and not force:
        w = np.load(cache)
        print(f"[Pesos] cargados de {cache} (huella {fp}, {n_files} máscaras): "
              f"{np.round(w, 3)}")
        return torch.from_numpy(w).float()
    print(f"[Pesos] escaneando máscaras para calcular pesos (huella {fp})...")
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    ds = LoveDADataset(dataset_root, split, 512, augment=False)
    for _, mp in ds.samples:
        m = np.array(Image.open(mp))
        if m.ndim == 3:
            m = m[:, :, 0]
        for c in range(NUM_CLASSES):
            counts[c] += np.sum(m == c)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    w = w / w.sum() * NUM_CLASSES          # normalización media
    np.save(cache, w)
    print(f"[Pesos] {np.round(w, 3)} guardados en {cache}")
    return torch.from_numpy(w).float()


# ───────────────────────── Métricas ─────────────────────────
def confusion_of(model, loader, device):
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = model(x).argmax(1).cpu().numpy()
            y = y.numpy()
            valid = y != IGNORE_INDEX
            yv, pv = y[valid], p[valid]
            conf += np.bincount(NUM_CLASSES * yv + pv, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return conf


def metrics_from_conf(conf):
    inter = np.diag(conf)
    union = conf.sum(1) + conf.sum(0) - inter
    iou = inter / np.maximum(union, 1)
    prec = inter / np.maximum(conf.sum(0), 1)
    rec = inter / np.maximum(conf.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-8)
    present = conf.sum(1) > 0
    miou = iou[present].mean()
    return miou, iou, f1


# ───────────────────────── Training ─────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=os.path.join("dataset", "loveda_remapped"))
    ap.add_argument("--out-dir", default=os.path.join("outputs"))
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-weights", action="store_true", help="no usar pesos de clase")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true",
                    help="cudnn.deterministic: más lento pero reproducible")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping en épocas sin mejora (0 = desactivado)")
    args = ap.parse_args(argv)

    # ⚠ ANTES: torch.manual_seed(42) y random.seed(42), pero NO np.random.seed
    #   ni cudnn.deterministic. Con augmentación que usa numpy y convoluciones en
    #   GPU, dos corridas idénticas daban resultados distintos. Sólo 6 de ~20
    #   scripts de entrenamiento fijaban semilla alguna.
    set_seed(args.seed, deterministic=args.deterministic)
    device = pick_device("cuda")
    print(f"[Device] {device} · seed {args.seed}"
          + (" · determinista" if args.deterministic else ""))
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = LoveDADataset(args.dataset_root, "Train", args.img_size, augment=True)
    val_ds = LoveDADataset(args.dataset_root, "Val", args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = DeepLabV3PlusMobileNetV2(NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Modelo] DeepLabV3+ MobileNetV2, {n_params:,} parámetros")

    weight = None if args.no_weights else class_weights(args.dataset_root).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_miou = 0.0
    sin_mejora = 0

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
        scheduler.step()

        conf = confusion_of(model, val_loader, device)
        miou, iou, f1 = metrics_from_conf(conf)
        print(f"Epoch {epoch+1:02d}/{args.epochs} | loss {run_loss/len(train_loader):.4f} | mIoU {miou:.4f}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {name:<12} IoU {iou[i]:.4f}  F1 {f1[i]:.4f}")

        if miou > best_miou:
            best_miou = miou
            sin_mejora = 0
            # Formato único de checkpoint (cansat/checkpoints.py). Antes train.py
            # guardaba un dict con metadata y los otros 15 scripts el state_dict
            # crudo, y ningún exportador servía para los dos formatos.
            from cansat.checkpoints import save_ckpt
            save_ckpt(os.path.join(args.out_dir, "best_model.pth"), model,
                      num_classes=NUM_CLASSES, img_size=args.img_size,
                      class_names=CLASS_NAMES, miou=float(miou),
                      iou_per_class=iou.tolist(), f1_per_class=f1.tolist(),
                      dataset=args.dataset_root, script="train.py",
                      epochs=epoch + 1, seed=args.seed,
                      batch_size=args.batch_size, lr=args.lr)
            np.save(os.path.join(args.out_dir, "best_confusion.npy"), conf)
            print(f"    → mejor checkpoint guardado (mIoU {miou:.4f})")
        else:
            sin_mejora += 1
            if args.patience and sin_mejora >= args.patience:
                print(f"\n[early stop] {sin_mejora} épocas sin mejorar "
                      f"(mejor mIoU {best_miou:.4f})")
                break

    print(f"\n[FIN] Mejor mIoU val: {best_miou:.4f}")
    print(f"Checkpoint: {os.path.join(args.out_dir, 'best_model.pth')}")
    print("⚠ No cites este número de memoria: quedó registrado en el checkpoint y")
    print("  en outputs/best_confusion.npy. Para el DPD usá evaluate.py, que")
    print("  escribe outputs/metrics/*.json con hash del modelo y tamaño de muestra.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
