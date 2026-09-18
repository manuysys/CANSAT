"""
CanSat La Base — Fase 4b: mejor modelo de daños (crops nativos xBD).
Extrae parches de 320×320 de las imágenes originales (sin downscaling)
para preservar detalle de techos.

⚠ CAMBIO 2026-09-17: split POR DESASTRE (antes era por fila y filtraba
  geografía: el IoU de val estaba inflado). La selección del mejor checkpoint
  ahora usa el IoU de daño sobre edificios (la métrica de la misión), no el
  mIoU de 3 clases dominado por el fondo.

Uso:
    python train_damage_v2.py --epochs 12
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from cansat.checkpoints import save_ckpt
from cansat.seed import set_seed
from cansat.xbd import iou_dano_edificios, split_por_desastre
from train import DeepLabV3PlusMobileNetV2

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASSES = ["other", "intacto", "dañado"]
MANIFEST = Path("dataset/xbd_masks/manifest.csv")


class XBDv2(Dataset):
    def __init__(self, rows, size=320, aug=False):
        self.rows, self.size, self.aug = rows, size, aug

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(r["image"])
        msk = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
        msk = np.where(msk >= 2, 2, msk).astype(np.uint8)   # 2-3-4 -> dañado

        # Crop nativo: extraer parche de 320×320 sin redimensionar
        h, w = img.shape[:2]
        if min(h, w) >= self.size:
            y0 = random.randint(0, h - self.size) if self.aug else (h - self.size) // 2
            x0 = random.randint(0, w - self.size) if self.aug else (w - self.size) // 2
            img = img[y0:y0 + self.size, x0:x0 + self.size]
            msk = msk[y0:y0 + self.size, x0:x0 + self.size]
        else:
            img = cv2.resize(img, (self.size, self.size))
            msk = cv2.resize(msk, (self.size, self.size),
                             interpolation=cv2.INTER_NEAREST)

        if self.aug:
            if random.random() < 0.5:
                img, msk = cv2.flip(img, 1), cv2.flip(msk, 1)
            if random.random() < 0.5:
                img, msk = cv2.flip(img, 0), cv2.flip(msk, 0)
            k = random.choice([0, 1, 2, 3])
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))
                msk = np.ascontiguousarray(np.rot90(msk, k))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        y = torch.from_numpy(msk.astype(np.int64))
        return x, y


def conf_matrix(pred, targ, n=3):
    pred = pred.ravel()
    targ = targ.ravel()
    ok = (targ >= 0) & (targ < n)
    t, p = targ[ok].astype(int), pred[ok].astype(int)
    return np.bincount(n * t + p, minlength=n * n).reshape(n, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="outputs/best_damage3.pth")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="fracción de DESASTRES reservados para val")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    with open(MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # ⚠ Split POR DESASTRE: el holdout son eventos completos que el modelo
    #   nunca ve en train. Antes era 90/10 por fila y filtraba geografía.
    train_rows, val_rows, val_grupos = split_por_desastre(
        rows, args.holdout_frac, args.seed)
    print(f"Muestras: {len(rows)} → train {len(train_rows)} / val {len(val_rows)}")
    print(f"Val por desastre: {val_grupos}")

    totals = np.zeros(3)
    for r in train_rows:
        totals[1] += int(r["intacto"])
        totals[2] += int(r["menor"]) + int(r["mayor"]) + int(r["destruido"])
    w = 1.0 / (totals + 1)
    w = w / w[1]
    w = np.clip(w, 0.2, 5)
    w[0] = 0.2
    wtorch = torch.tensor(w, dtype=torch.float32)
    print(f"Pesos por clase: {np.round(w, 2).tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    try:
        model = DeepLabV3PlusMobileNetV2(3)
    except TypeError:
        model = DeepLabV3PlusMobileNetV2()
    ckpt = Path("outputs/best_damage3.pth")
    if ckpt.exists():
        # load_into acepta los dos formatos y aborta si la cobertura es baja
        # (ver train_damage_v3.py: esto ya falló en silencio una vez).
        from cansat.checkpoints import load_into
        _, frac = load_into(model, ckpt, strict=False, min_loaded_frac=0.5,
                            verbose=True)
        print(f"Checkpoint damage3 cargado ({frac:.0%} de las capas)")
    model = model.to(device)
    wtorch = wtorch.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_dl = DataLoader(XBDv2(train_rows, args.size, aug=True),
                          batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(XBDv2(val_rows, args.size), batch_size=args.batch)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        for bi, (xb, yb) in enumerate(train_dl):
            x, y = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=wtorch)
            loss.backward()
            opt.step()
            if bi % 50 == 0:
                print(f"    batch {bi}/{len(train_dl)} loss {loss.item():.4f}")
        print(f"  epoch {ep + 1}: loss {loss.item():.4f} — evaluando...")

        model.eval()
        cm = np.zeros((3, 3), dtype=np.int64)
        iou_edif: list[float] = []
        with torch.no_grad():
            for x, y in val_dl:
                y_np = y.numpy()
                pred = model(x.to(device)).argmax(1).cpu().numpy()
                cm += conf_matrix(pred, y_np)
                for pr, gt in zip(pred, y_np, strict=False):
                    iou_edif.append(iou_dano_edificios(pr, gt))
        ious = []
        for i in range(3):
            inter = cm[i, i]
            union = cm[i, :].sum() + cm[:, i].sum() - inter
            ious.append(inter / union if union else 0.0)
        miou = float(np.mean(ious))
        # Métrica de selección = la de la misión (daño sobre edificios), no el
        # mIoU de 3 clases donde el fondo domina.
        sel = float(np.mean(iou_edif)) if iou_edif else 0.0
        print(f"  IoU: other={ious[0]:.2f}  intacto={ious[1]:.2f}  "
              f"DAÑADO={ious[2]:.2f}  | mIoU={miou:.3f}  | "
              f"DAÑADO* (edificios)={sel:.3f}")
        if sel > best:
            best = sel
            save_ckpt(args.out, model,
                      num_classes=3, img_size=args.size, class_names=CLASSES,
                      iou_dano_edificios=round(sel, 4),
                      miou_3clases=round(miou, 4),
                      iou_per_class=[round(float(x), 4) for x in ious],
                      dataset="xbd (split por desastre)",
                      val_desastres=val_grupos,
                      script="train_damage_v2.py",
                      epochs=ep + 1, seed=args.seed)
            print(f"  → guardado {args.out}")

    print(f"[OK] mejor DAÑADO* (edificios): {best:.3f}")


if __name__ == "__main__":
    main()
