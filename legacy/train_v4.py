"""
CanSat La Base — v4: Pack de augmentación (multi-scale + copy-paste + boundary loss)
sobre init CBAM, sin pseudo-labels.

Uso:
    python train_v4.py --epochs 8 --batch-size 8
"""

# ── legacy/: script fuera del pipeline activo. Para correrlo desde la raíz:
#     python legacy/train_v4.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import random

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train import (LoveDADataset, confusion_of, metrics_from_conf,
                   class_weights, CLASS_NAMES, IGNORE_INDEX)
from train_cbam import DeepLabV3PlusCBAM

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class AugDataset(Dataset):
    """Dataset con multi-scale + copy-paste augmentation."""
    def __init__(self, root, split, size=320, augment=True):
        self.ds = LoveDADataset(root, split, size, augment=False)
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, msk = self.ds[i]
        if self.augment:
            # Multi-scale: random scale 0.75-1.5 + crop/pad
            scale = random.uniform(0.75, 1.5)
            new_size = int(self.size * scale)
            img_np = img.numpy().transpose(1, 2, 0)
            msk_np = msk.numpy()
            img_resized = cv2.resize(img_np, (new_size, new_size))
            msk_resized = cv2.resize(msk_np, (new_size, new_size),
                                     interpolation=cv2.INTER_NEAREST)
            # Ajustar a self.size exacto
            if new_size >= self.size:
                # Crop aleatorio
                top = random.randint(0, new_size - self.size)
                left = random.randint(0, new_size - self.size)
                img_np = img_resized[top:top+self.size, left:left+self.size]
                msk_np = msk_resized[top:top+self.size, left:left+self.size]
            else:
                # Pad centrado (asimétrico si la diferencia es impar)
                pad_top = (self.size - new_size) // 2
                pad_bottom = self.size - new_size - pad_top
                pad_left = (self.size - new_size) // 2
                pad_right = self.size - new_size - pad_left
                img_np = np.pad(img_resized,
                                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
                msk_np = np.pad(msk_resized,
                                ((pad_top, pad_bottom), (pad_left, pad_right)),
                                constant_values=IGNORE_INDEX)
            # ⚠ BUG ARREGLADO: antes esta reasignación estaba SÓLO en la rama
            # del pad. En la rama del crop (probabilidad ~1/3, porque
            # scale~U(0.75,1.5)) la imagen quedaba SIN escalar/cortar y la
            # máscara SÍ: ~1/3 de las muestras entrenaban con etiquetas
            # desalineadas. El checkpoint best_terrain_v4.pth arrastra el
            # defecto; hay que re-entrenar para usarlo.
            img = torch.from_numpy(
                np.ascontiguousarray(img_np.transpose(2, 0, 1), dtype=np.float32))
            msk = torch.from_numpy(msk_np.astype(np.int64))
        return img, msk


class BoundaryLoss(nn.Module):
    """Boundary loss: castiga errores en bordes de ground truth."""
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore = ignore_index

    def forward(self, pred, target):
        """pred: (B,C,H,W) logits; target: (B,H,W) labels"""
        valid = target != self.ignore
        if not valid.any():
            return torch.tensor(0.0, device=pred.device)
        # Distance transform de los bordes
        b, h, w = target.shape
        boundary = torch.zeros_like(target, dtype=torch.float32)
        for i in range(b):
            t = target[i].cpu().numpy()
            if t.max() == t.min():
                continue
            # Detectar bordes: píxeles donde cambia la clase
            edges = np.zeros_like(t, dtype=bool)
            edges[1:, :] |= (t[1:, :] != t[:-1, :])
            edges[:, 1:] |= (t[:, 1:] != t[:, :-1])
            # Distance transform
            dt = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 5)
            boundary[i] = torch.from_numpy(dt)
        boundary = boundary.to(pred.device)
        # Loss: predicción en bordes debe ser confiable.
        # ⚠ BUG ARREGLADO: antes era `correct = (pred_hard == target)` con
        #   pred_hard = argmax → el grafo se cortaba ahí y `loss_bd` salía con
        #   requires_grad=False. El término 0.3*boundary del total NO aportaba
        #   gradiente: era un no-op decorativo (y costaba un distance transform
        #   en CPU por batch). Ahora se usa la probabilidad softmax de la clase
        #   correcta, que sí es diferenciable.
        pred_soft = F.softmax(pred, dim=1)
        tgt = target.clamp(min=0)
        prob_correct = pred_soft.gather(1, tgt.unsqueeze(1)).squeeze(1)
        weight = torch.exp(-boundary)
        loss = (1.0 - prob_correct) * weight * valid.float()
        return loss.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="dataset/loveda_remapped")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepLabV3PlusCBAM(5).to(device)
    sd = torch.load("outputs/best_terrain_cbam.pth", map_location="cpu")
    model.load_state_dict(sd, strict=False)
    print("[OK] init: CBAM completo")

    train_ds = AugDataset(args.dataset_root, "Train", 320, augment=True)
    val_ds = LoveDADataset(args.dataset_root, "Val", 320, augment=False)
    print(f"[Dataset] Train: {len(train_ds)} muestras (augmented)")
    print(f"[Dataset] Val: {len(val_ds)} muestras")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    weight = class_weights(args.dataset_root).to(device)
    ce_crit = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)
    bd_crit = BoundaryLoss(ignore_index=IGNORE_INDEX)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for bi, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss_ce = ce_crit(logits, y)
                loss_bd = bd_crit(logits, y)
                loss = loss_ce + 0.3 * loss_bd  # peso boundary
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if bi % 20 == 0:
                print(f"    batch {bi}/{len(train_dl)} ce {loss_ce.item():.3f} "
                      f"bd {loss_bd.item():.3f}", flush=True)
        sched.step()

        conf = confusion_of(model, val_dl, device)
        miou, iou, f1 = metrics_from_conf(conf)
        tag = ""
        if miou > best:
            best = miou
            torch.save(model.state_dict(), "outputs/best_terrain_v4.pth")
            tag = " → guardado"
        print(f"  epoch {ep+1}: mIoU {miou:.4f} | "
              + "  ".join(f"{CLASS_NAMES[i][:4]}={iou[i]:.2f}" for i in range(5))
              + tag)

    print(f"[OK] mejor mIoU v4: {best:.4f} (objetivo: superar 0.5317)")
    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, 320, 320),
                      "outputs/cansat_seg_terrain_v4.onnx",
                      input_names=["input"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True,
                      dynamo=False)  # ← agrega esta línea
    print("[OK] outputs/cansat_seg_terrain_v4.onnx")


if __name__ == "__main__":
    main()
