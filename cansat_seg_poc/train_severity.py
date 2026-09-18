"""
CanSat La Base — F2b: severidad del daño (5 clases) para estimar pérdidas.

El DPD pide estimar las pérdidas humanas "según su magnitud". Hasta ahora
`cansat/casualties.py` usaba una fracción de colapso FIJA (0.3). RescueNet trae
los 4 niveles de daño edilicio, así que se entrena un modelo de severidad:

    0 = other (fondo, agua, rutas, árboles…)
    1 = edificio sin daño
    2 = daño menor
    3 = daño mayor
    4 = destrucción total

La fracción de colapso medida por frame es
    colapso = (mayor + destruido) / (cualquier edificio)
y alimenta `casualties.estimar(..., collapse_frac_medido=...)`.

⚠ Los tiles ya generados tienen el daño colapsado a una sola clase; este script
  NO regenera nada: reconstruye el recorte desde la máscara ORIGINAL de
  RescueNet usando las coordenadas que quedaron en el nombre del tile
  (`rescuenet_<split>_<id>_<y0>_<x0>`).

Uso:
    python train_severity.py --epochs 10 --manifest dataset/rescuenet_tiles/manifest_train_sub4000.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cansat.checkpoints import load_into, save_ckpt
from cansat.seed import set_seed
from train import DeepLabV3PlusMobileNetV2

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RAW = Path("dataset/rescuenet")
TILE = 640
CLASSES = ["other", "intacto", "menor", "mayor", "destruido"]
# RescueNet original → severidad (5 clases del modelo)
REMAP = {2: 1, 3: 2, 4: 3, 5: 4}


def parsear_nombre(name: str) -> tuple[str, str, int, int]:
    """``rescuenet_train_10781_2360_640`` → (split, id, y0, x0)."""
    parts = name.split("_")
    return parts[1], parts[2], int(parts[3]), int(parts[4])


class RescueNetSeverity(Dataset):
    def __init__(self, manifest: Path, size: int = 320, aug: bool = False):
        self.size, self.aug = size, aug
        with manifest.open(encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        split, sid, y0, x0 = parsear_nombre(r["name"])
        img = cv2.imread(str(RAW / f"{split}-org-img" / f"{sid}.jpg"))
        lab = cv2.imread(str(RAW / f"{split}-label-img" / f"{sid}_lab.png"),
                         cv2.IMREAD_GRAYSCALE)
        img = img[y0:y0 + TILE, x0:x0 + TILE]
        lab = lab[y0:y0 + TILE, x0:x0 + TILE]
        msk = np.zeros_like(lab, dtype=np.uint8)
        for src, dst in REMAP.items():
            msk[lab == src] = dst
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
        return (torch.from_numpy(rgb.transpose(2, 0, 1)).float(),
                torch.from_numpy(msk.astype(np.int64)))


def compute_weights(ds: RescueNetSeverity, n=5, clip=(0.3, 8.0)):
    hist = np.zeros(n, dtype=np.float64)
    for r in ds.rows:
        split, sid, y0, x0 = parsear_nombre(r["name"])
        lab = cv2.imread(str(RAW / f"{split}-label-img" / f"{sid}_lab.png"),
                         cv2.IMREAD_GRAYSCALE)
        if lab is None:
            continue
        lab = lab[y0:y0 + TILE, x0:x0 + TILE]
        msk = np.zeros_like(lab, dtype=np.uint8)
        for src, dst in REMAP.items():
            msk[lab == src] = dst
        hist += np.bincount(msk.ravel(), minlength=n)[:n]
    freq = hist / max(1.0, hist.sum())
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-9))
    w = np.clip(w, clip[0], clip[1])
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32), freq


def conf_matrix(pred, tgt, n=5):
    idx = tgt.ravel().astype(np.int64) * n + pred.ravel().astype(np.int64)
    return np.bincount(idx, minlength=n * n).reshape(n, n).astype(np.int64)


def evaluar(model, loader, device):
    """Devuelve (cm, IoU colapso sobre edificios) — la métrica que usa casualties."""
    cm = np.zeros((5, 5), dtype=np.int64)
    inter_c = union_c = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            yt = y.numpy()
            cm += conf_matrix(pred, yt)
            pc = pred >= 3          # mayor + destruido predichos
            gc = yt >= 3            # mayor + destruido GT
            gb = yt >= 1            # cualquier edificio GT
            inter_c += int((pc & gc).sum())
            union_c += int(((pc & gb) | gc).sum())
    return cm, (inter_c / union_c if union_c else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrena severidad del daño (5 clases)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--manifest", default="dataset/rescuenet_tiles/manifest_train_sub4000.csv")
    ap.add_argument("--val-manifest", default="dataset/rescuenet_tiles/manifest_val.csv")
    ap.add_argument("--out", default="outputs/best_severity.pth")
    ap.add_argument("--onnx-out", default="outputs/cansat_severity.onnx")
    ap.add_argument("--init", default="outputs/best_damage_v3_bal.pth")
    ap.add_argument("--resume", action="store_true",
                    help="continuar desde --out si ya existe (más épocas)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    train_ds = RescueNetSeverity(Path(args.manifest), args.size, aug=True)
    val_ds = RescueNetSeverity(Path(args.val_manifest), args.size)
    print(f"Pares: train {len(train_ds)} / val {len(val_ds)} · {args.size}px")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3PlusMobileNetV2(5)
    init_path = (args.out if (args.resume and Path(args.out).is_file())
                 else args.init)
    if init_path and Path(init_path).is_file():
        _, frac = load_into(model, init_path, strict=False,
                            min_loaded_frac=0.3, verbose=False)
        print(f"Init desde {init_path}: {frac:.0%} de las capas")
    model = model.to(device)

    w, freq = compute_weights(train_ds)
    print("Frecuencias train: " + ", ".join(f"{f * 100:.2f}%" for f in freq))
    print("Pesos de clase   : " + ", ".join(f"{v:.2f}" for v in w.tolist()))
    w = w.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch,
                        num_workers=args.workers,
                        persistent_workers=args.workers > 0)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for bi, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=w)
            loss.backward()
            opt.step()
            run += loss.item()
            if bi % 50 == 0:
                print(f"    batch {bi}/{len(train_dl)} loss {loss.item():.4f}")
        sched.step()

        model.eval()
        cm, iou_col = evaluar(model, val_dl, device)
        ious = []
        for i in range(5):
            inter = cm[i, i]
            union = cm[i, :].sum() + cm[:, i].sum() - inter
            ious.append(inter / union if union else 0.0)
        print(f"  epoch {ep + 1}: " +
              " ".join(f"{CLASSES[i]}={ious[i]:.2f}" for i in range(5)) +
              f" | COLAPSO*={iou_col:.3f} | loss {run / len(train_dl):.4f}")
        if iou_col > best:
            best = iou_col
            save_ckpt(args.out, model, num_classes=5, img_size=args.size,
                      class_names=CLASSES,
                      iou_per_class=[round(x, 4) for x in ious],
                      iou_colapso_edificios=round(iou_col, 4),
                      dataset="RescueNet (severidad original)",
                      script="train_severity.py",
                      epochs=ep + 1, seed=args.seed)
            print(f"  → guardado {args.out}")

    print(f"[OK] mejor IoU COLAPSO* (mayor+destruido sobre edificios): {best:.3f}")

    from cansat.checkpoints import load_model_state
    model.load_state_dict(load_model_state(args.out, strict=True))
    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, args.size, args.size),
                      args.onnx_out,
                      input_names=["input"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True,
                      dynamo=False)
    print(f"[OK] {args.onnx_out} (BEST, dynamo off)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
