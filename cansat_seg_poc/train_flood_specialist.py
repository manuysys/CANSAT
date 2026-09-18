"""
CanSat La Base — Especialista de inundación (FloodNet).
3 clases: other / flood / agua-normal.

⚠ Los pesos actuales del repo eran INSERVIBLES: prepare_floodnet.py desplazaba
  las clases con un `shift` heurístico, así que el modelo aprendió "flood" =
  Building-non-flooded y "agua normal" = Tree (ver el comentario largo en
  prepare_floodnet.py). Con el remapeo corregido hay que RE-ENTRENAR sí o sí.

Uso:
    python prepare_floodnet.py            # dataset remapeado (ya corregido)
    python train_flood_specialist.py --epochs 12
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cansat.checkpoints import save_ckpt
from cansat.seed import set_seed
from train import DeepLabV3PlusMobileNetV2


def conf_matrix(pred, tgt, n=3):
    """Matriz de confusión vectorizada (bincount). Antes era doble loop Python."""
    idx = tgt.ravel().astype(np.int64) * n + pred.ravel().astype(np.int64)
    return np.bincount(idx, minlength=n * n).reshape(n, n).astype(np.int64)


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ROOT = Path("dataset/floodnet_remapped")


class FloodDS(Dataset):
    def __init__(self, split, size=320, aug=False):
        self.size, self.aug = size, aug
        ib, mb = ROOT / split / "images", ROOT / split / "masks"
        self.pairs = [(ip, mb / ip.name) for ip in sorted(ib.glob("*.png"))
                      if (mb / ip.name).exists()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.imread(str(ip))
        msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
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


def compute_weights(ds, n=3, clip=(0.3, 3.0)):
    """
    Pesos inversos a la raíz de la frecuencia, normalizados a media 1.

    Antes eran `[0.3, 1.0, 1.0]` hardcodeados: con el dataset corregido la
    clase flood pasó a ser el 3.5 % de los píxeles y el peso fijo la dejaba
    sin señal. Se calculan del split de TRAIN (nunca de val).
    """
    hist = np.zeros(n, dtype=np.float64)
    for ip, mp in ds.pairs:
        g = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        hist += np.bincount(g.ravel(), minlength=n)[:n]
    freq = hist / max(1.0, hist.sum())
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-9))
    w = np.clip(w, clip[0], clip[1])
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32), freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--out", default="outputs/best_flood_specialist.pth")
    ap.add_argument("--onnx-out", default="outputs/cansat_flood_specialist.onnx")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    train_ds = FloodDS("train", size=args.size, aug=True)
    val_ds = FloodDS("val", size=args.size)
    print(f"Pares: train {len(train_ds)} / val {len(val_ds)} · {args.size}px")
    if not train_ds:
        print("[ERROR] Corré antes prepare_floodnet.py")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3PlusMobileNetV2(3).to(device)
    w, freq = compute_weights(train_ds)
    print("Frecuencias train: " + ", ".join(f"{f * 100:.2f}%" for f in freq))
    print("Pesos de clase   : " + ", ".join(f"{v:.2f}" for v in w.tolist()))
    w = w.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch,
                        num_workers=args.workers,
                        persistent_workers=args.workers > 0)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        for bi, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=w)
            loss.backward()
            opt.step()
            if bi % 30 == 0:
                print(f"    batch {bi}/{len(train_dl)} loss {loss.item():.4f}")
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
        print(f"  epoch {ep + 1}: other={ious[0]:.2f} FLOOD={ious[1]:.2f} "
              f"agua={ious[2]:.2f}")
        if ious[1] > best:
            best = ious[1]
            save_ckpt(args.out, model,
                      num_classes=3, img_size=args.size,
                      class_names=["other", "flood", "agua_normal"],
                      iou_per_class=[round(x, 4) for x in ious],
                      iou_flood=round(ious[1], 4),
                      dataset="floodnet_remapped (remapeo corregido)",
                      script="train_flood_specialist.py",
                      epochs=args.epochs, seed=args.seed)
            print("  → guardado")
    print(f"[OK] mejor IoU FLOOD: {best:.3f}")

    # Exportar EL MEJOR checkpoint (antes se exportaba el modelo en memoria de
    # la última época) y con dynamo=False: el default de torch 2.11 genera IR10
    # con pesos externos, que ni cv2.dnn ni el conversor IMX500 aceptan.
    from cansat.checkpoints import load_model_state
    model.load_state_dict(load_model_state(args.out, strict=True))
    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, args.size, args.size),
                      args.onnx_out,
                      input_names=["input"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True,
                      dynamo=False)
    print(f"[OK] {args.onnx_out} (modelo BEST, dinamo off)")
    print(f"     Verificá: python audit_imx500.py --onnx {args.onnx_out}")


if __name__ == "__main__":
    main()
