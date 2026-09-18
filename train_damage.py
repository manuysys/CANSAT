"""
CanSat La Base — Fine-tuning del modelo de daños con xBD (fase D2).
Clases: 0=other/fondo, 1=intacto, 2=daño_menor, 3=daño_mayor, 4=destruido.
Arranca desde el checkpoint de LoveDA (transfer learning).

Uso:
    python train_damage.py --max-samples 600 --epochs 3   # prueba rápida
    python train_damage.py --epochs 8                     # corrida seria
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

from train import DeepLabV3PlusMobileNetV2

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASSES = ["other", "intacto", "daño_menor", "daño_mayor", "destruido"]


class XBDDataset(Dataset):
    def __init__(self, rows, size=512, aug=False):
        self.rows, self.size, self.aug = rows, size, aug

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(r["image"])
        img = cv2.resize(img, (self.size, self.size))
        msk = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
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


def conf_matrix(pred, targ, n=5):
    ok = (targ >= 0) & (targ < n)
    t, p = targ[ok].astype(int), pred[ok].astype(int)
    return np.bincount(n * t + p, minlength=n * n).reshape(n, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset/xbd_masks/manifest.csv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="outputs/best_damage.pth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # Semilla completa (random+numpy+torch+cudnn): este script no fijaba
    # NINGUNA y la augmentacion usa el modulo random global.
    from cansat.seed import set_seed
    set_seed(getattr(args, "seed", 42))


    with open(args.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # ⚠ Split POR DESASTRE (ver cansat/xbd.py): el 90/10 por fila de antes
    #   repartía tiles vecinos del mismo evento entre train y val.
    from cansat.xbd import split_por_desastre
    train_rows, val_rows, val_grupos = split_por_desastre(rows, 0.2, args.seed)
    if args.max_samples:
        train_rows = train_rows[:args.max_samples]
    print(f"Muestras: {len(rows)} → train {len(train_rows)} / val {len(val_rows)}")
    print(f"Val por desastre: {val_grupos}")

    # Pesos por clase (compensa desbalance)
    totals = np.zeros(5)
    for r in train_rows:
        totals[1] += int(r["intacto"])
        totals[2] += int(r["menor"])
        totals[3] += int(r["mayor"])
        totals[4] += int(r["destruido"])
    w = 1.0 / np.sqrt(totals + 1)      # raíz cuadrada: suaviza el desbalance
    w = w / w[1]                       # intacto = 1.0
    w = np.clip(w, 0.2, 5)             # ninguna clase domina ni se sacrifica
    w[0] = 0.2                         # fondo: es mayoría, peso chico
    wtorch = torch.tensor(w, dtype=torch.float32)
    print(f"Pesos por clase: {np.round(w, 2).tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    try:
        model = DeepLabV3PlusMobileNetV2(5)
    except TypeError:
        model = DeepLabV3PlusMobileNetV2()
    ckpt = Path("outputs/best_model.pth")
    if ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=False)
        print("Checkpoint LoveDA cargado (transfer learning)")
    model = model.to(device)
    wtorch = wtorch.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_dl = DataLoader(XBDDataset(train_rows, args.size, aug=True),
                          batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(XBDDataset(val_rows, args.size),
                        batch_size=args.batch)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        for bi, (xb, yb) in enumerate(train_dl):
            x, y = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=wtorch)
            loss.backward()
            opt.step()
            if bi % 10 == 0:
                print(f"    batch {bi}/{len(train_dl)} loss {loss.item():.4f}")
        print(f"  epoch {ep + 1}: loss {loss.item():.4f} — evaluando...")

        model.eval()
        cm = np.zeros((5, 5), dtype=np.int64)
        bin_inter = bin_union = 0
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x.to(device)).argmax(1).cpu().numpy()
                cm += conf_matrix(pred, y.numpy())
                pd, td = (pred >= 2), (y.numpy() >= 2)
                bin_inter += int((pd & td).sum())
                bin_union += int((pd | td).sum())
        iou_dano = bin_inter / bin_union if bin_union else 0.0
        ious = []
        for i in range(5):
            inter = cm[i, i]
            union = cm[i, :].sum() + cm[:, i].sum() - inter
            ious.append(inter / union if union else 0.0)
        miou = float(np.mean(ious))
        print("  IoU por clase: " + "  ".join(
            f"{CLASSES[i]}={ious[i]:.2f}" for i in range(5))
            + f"  | DAÑO(any)={iou_dano:.2f}")
        print(f"  mIoU daños   : {miou:.3f}")
        if miou > best:
            best = miou
            torch.save(model.state_dict(), args.out)
            print(f"  → guardado {args.out}")

    print(f"[OK] mejor mIoU: {best:.3f}")


if __name__ == "__main__":
    main()
