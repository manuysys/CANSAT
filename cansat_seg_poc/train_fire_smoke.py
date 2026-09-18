"""
CanSat La Base — F3: detector de fuego/humo (3 clases: other/fuego/humo).

Entrena sobre `dataset/fire_smoke` (LibreYOLO/fire-smoke-seg, FLAME, CC BY 4.0)
inicializando el encoder desde el modelo de terreno v2 (dominio aéreo) y
exporta a ONNX opset 17 autocontenido para el pipeline (fire_pct / smoke_pct).

Uso:
    python prepare_fire_smoke.py
    python train_fire_smoke.py --epochs 30
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
ROOT = Path("dataset/fire_smoke")
CLASSES = ["other", "fuego", "humo"]


class FireSmokeDS(Dataset):
    def __init__(self, split: str, size: int = 256, aug: bool = False):
        self.size, self.aug = size, aug
        man = ROOT / f"manifest_{split}.csv"
        with man.open(encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(r["image"])
        msk = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
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


def compute_weights(ds: FireSmokeDS, n=3, clip=(0.3, 6.0)):
    hist = np.zeros(n, dtype=np.float64)
    for r in ds.rows:
        g = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        hist += np.bincount(g.ravel(), minlength=n)[:n]
    freq = hist / max(1.0, hist.sum())
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-9))
    w = np.clip(w, clip[0], clip[1])
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32), freq


def conf_matrix(pred, tgt, n=3):
    idx = tgt.ravel().astype(np.int64) * n + pred.ravel().astype(np.int64)
    return np.bincount(idx, minlength=n * n).reshape(n, n).astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrena fuego/humo (3 clases)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="outputs/best_fire_smoke.pth")
    ap.add_argument("--onnx-out", default="outputs/cansat_fire_smoke.onnx")
    ap.add_argument("--init", default="outputs/best_terrain_v2.pth",
                    help="checkpoint para inicializar (dominio aéreo); '' = nada")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    train_ds = FireSmokeDS("train", args.size, aug=True)
    val_ds = FireSmokeDS("valid", args.size)
    if not len(train_ds):
        print("[ERROR] Corré antes prepare_fire_smoke.py")
        return 1
    print(f"Pares: train {len(train_ds)} / val {len(val_ds)} · {args.size}px")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3PlusMobileNetV2(3)
    if args.init and Path(args.init).is_file():
        _, frac = load_into(model, args.init, strict=False,
                            min_loaded_frac=0.3, verbose=False)
        print(f"Init desde {args.init}: {frac:.0%} de las capas "
              f"(encoder aéreo transferido)")
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
            if bi % 10 == 0:
                print(f"    batch {bi}/{len(train_dl)} loss {loss.item():.4f}")
        sched.step()

        model.eval()
        cm = np.zeros((3, 3), dtype=np.int64)
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x.to(device)).argmax(1).cpu().numpy()
                cm += conf_matrix(pred, y.numpy())
        ious = []
        for i in range(3):
            inter = cm[i, i]
            union = cm[i, :].sum() + cm[:, i].sum() - inter
            ious.append(inter / union if union else 0.0)
        sel = (ious[1] + ious[2]) / 2.0     # fuego y humo pesan igual
        print(f"  epoch {ep + 1}: other={ious[0]:.2f} FUEGO={ious[1]:.2f} "
              f"HUMO={ious[2]:.2f} | media={sel:.3f} | loss {run / len(train_dl):.4f}")
        if sel > best:
            best = sel
            save_ckpt(args.out, model, num_classes=3, img_size=args.size,
                      class_names=CLASSES,
                      iou_per_class=[round(x, 4) for x in ious],
                      iou_media_fuego_humo=round(sel, 4),
                      dataset="fire_smoke_seg (FLAME, CC BY 4.0)",
                      script="train_fire_smoke.py",
                      epochs=ep + 1, seed=args.seed)
            print(f"  → guardado {args.out}")

    print(f"[OK] mejor media fuego/humo: {best:.3f}")

    from cansat.checkpoints import load_model_state
    model.load_state_dict(load_model_state(args.out, strict=True))
    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, args.size, args.size),
                      args.onnx_out,
                      input_names=["input"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True,
                      dynamo=False)
    print(f"[OK] {args.onnx_out} (BEST, dynamo off)")
    print(f"     Verificá: python audit_imx500.py --onnx {args.onnx_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
