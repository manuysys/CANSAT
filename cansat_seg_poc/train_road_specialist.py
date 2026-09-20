"""
CanSat La Base — Especialista de VÍAS (consulta terrestre).

2 clases: 0 = no-vía, 1 = vía. Datos de ``prepare_vias.py``: FloodNet
(road-flooded/non-flooded, UAV/inundación) + LoveDA raw (road, satelital 0.3 m).

⚠ GATE DE HONESTIDAD: el checkpoint se elige por IoU de vía en FloodNet val
(las MISMAS 80 imágenes que el val del especialista de inundación). Si el mejor
IoU no llega a ``UMBRAL_GATE``, la consulta de vías queda "no soportada" y se
documenta como técnica que no funcionó — no se integra al pipeline. LoveDA val
se reporta como dominio distinto, sin gate.

Uso:
    python prepare_vias.py
    python train_road_specialist.py --epochs 12
    python train_road_specialist.py --dominios floodnet   # variante sin LoveDA
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cansat.checkpoints import save_ckpt
from cansat.seed import set_seed
from train import DeepLabV3PlusMobileNetV2

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ROOT = Path("dataset/vias")
IGNORE = 255

#: IoU mínimo de vía en FloodNet val para que la consulta se considere soportada.
UMBRAL_GATE: float = 0.50


class ViasDS(Dataset):
    """Pares (imagen, máscara binaria 0/1 con 255=ignore) del manifest."""

    def __init__(self, rows: list[dict], size: int = 320, aug: bool = False):
        self.rows, self.size, self.aug = rows, size, aug

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = cv2.imread(r["image"])
        msk = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise FileNotFoundError(f"par ilegible: {r['image']} / {r['mask']}")
        s = self.size
        if self.aug:
            if random.random() < 0.5:
                img, msk = cv2.flip(img, 1), cv2.flip(msk, 1)
            if random.random() < 0.5:
                img, msk = cv2.flip(img, 0), cv2.flip(msk, 0)
            k = random.choice([0, 1, 2, 3])
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))
                msk = np.ascontiguousarray(np.rot90(msk, k))
        if img.shape[0] != s or img.shape[1] != s:
            img = cv2.resize(img, (s, s))
            msk = cv2.resize(msk, (s, s), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        return (torch.from_numpy(rgb.transpose(2, 0, 1)).float(),
                torch.from_numpy(msk.astype(np.int64)))


def leer_manifest(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_weights(ds: ViasDS, n: int = 2, clip=(0.3, 3.0)):
    hist = np.zeros(n, dtype=np.float64)
    for r in ds.rows:
        g = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        valid = g[g < n]
        hist += np.bincount(valid.ravel(), minlength=n)[:n]
    freq = hist / max(1.0, hist.sum())
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-9))
    w = np.clip(w, clip[0], clip[1])
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32), freq


def conf_matrix(pred: np.ndarray, targ: np.ndarray, n: int = 2) -> np.ndarray:
    pred, targ = pred.ravel(), targ.ravel()
    ok = (targ >= 0) & (targ < n)
    return np.bincount(n * targ[ok].astype(int) + pred[ok].astype(int),
                       minlength=n * n).reshape(n, n)


def iou_via(cm: np.ndarray) -> float:
    inter = cm[1, 1]
    union = cm[1, :].sum() + cm[:, 1].sum() - inter
    return float(inter / union) if union else 0.0


def evaluar(model, dl, device) -> tuple[float, np.ndarray]:
    model.eval()
    cm = np.zeros((2, 2), dtype=np.int64)
    with torch.no_grad():
        for x, y in dl:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            cm += conf_matrix(pred, y.numpy())
    return iou_via(cm), cm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dominios", default="floodnet,loveda",
                    help="dominios de entrenamiento separados por coma")
    ap.add_argument("--manifest-train", default="dataset/vias/manifest_train.csv")
    ap.add_argument("--manifest-val", default="dataset/vias/manifest_val.csv")
    ap.add_argument("--out", default="outputs/best_vias.pth")
    ap.add_argument("--onnx-out", default="outputs/cansat_vias.onnx")
    args = ap.parse_args()

    set_seed(args.seed)
    dominios = [d.strip() for d in args.dominios.split(",") if d.strip()]
    rows_train = leer_manifest(args.manifest_train)
    rows_val = leer_manifest(args.manifest_val)
    # Los manifests no traen columna split: train viene de manifest_train y val
    # de manifest_val. ``filtrar`` filtra por dominio y por split etiquetado.
    train_rows = [r for r in rows_train if r["dominio"] in dominios]
    val_rows = [r for r in rows_val if r["dominio"] in dominios]
    if not train_rows:
        print("[ERROR] sin filas de train. Corré prepare_vias.py.")
        return 1

    train_ds = ViasDS(train_rows, size=args.size, aug=True)
    val_fl = ViasDS([r for r in val_rows if r["dominio"] == "floodnet"], size=args.size)
    val_ld = ViasDS([r for r in val_rows if r["dominio"] == "loveda"], size=args.size)
    print(f"Dominios: {dominios} · train {len(train_ds)} · "
          f"val floodnet {len(val_fl)} · val loveda {len(val_ld)} · {args.size}px")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = DeepLabV3PlusMobileNetV2(2).to(device)
    w, freq = compute_weights(train_ds)
    print("Frecuencias train: " + ", ".join(f"{f * 100:.2f}%" for f in freq))
    print("Pesos de clase   : " + ", ".join(f"{v:.2f}" for v in w.tolist()))
    w = w.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers,
                          persistent_workers=args.workers > 0)
    fl_dl = DataLoader(val_fl, batch_size=args.batch, num_workers=0)
    ld_dl = DataLoader(val_ld, batch_size=args.batch, num_workers=0)

    best = -1.0
    best_ld = 0.0
    for ep in range(args.epochs):
        model.train()
        for bi, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            if not (y != IGNORE).any():
                continue
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=w, ignore_index=IGNORE)
            loss.backward()
            opt.step()
            if bi % 50 == 0:
                print(f"    batch {bi}/{len(train_dl)} loss {loss.item():.4f}",
                      flush=True)
        iou_fl, _ = evaluar(model, fl_dl, device)
        iou_ld, _ = evaluar(model, ld_dl, device) if len(ld_dl) else (0.0, None)
        print(f"  epoch {ep + 1}: IoU via floodnet={iou_fl:.3f} "
              f"loveda={iou_ld:.3f}", flush=True)
        if iou_fl > best:
            best, best_ld = iou_fl, iou_ld
            save_ckpt(args.out, model,
                      num_classes=2, img_size=args.size,
                      class_names=["no_via", "via"],
                      iou_via_floodnet=round(iou_fl, 4),
                      iou_via_loveda=round(iou_ld, 4),
                      gate_umbral=UMBRAL_GATE,
                      gate_pasa=bool(iou_fl >= UMBRAL_GATE),
                      dataset=f"vias: FloodNet+LoveDA ({'+'.join(dominios)})",
                      script="train_road_specialist.py",
                      epochs=args.epochs, seed=args.seed)
            print("  → guardado", flush=True)

    print(f"[OK] mejor IoU via (FloodNet val): {best:.3f} · LoveDA val: {best_ld:.3f}")
    if best >= UMBRAL_GATE:
        print(f"[OK] PASA el gate (>= {UMBRAL_GATE}): la consulta de vías se integra.")
        exportar(model, args.out, args.onnx_out, args.size)
    else:
        print(f"[WARN] NO PASA el gate (< {UMBRAL_GATE}): la consulta de vías "
              f"queda NO SOPORTADA y se documenta como técnica que no funcionó. "
              f"No se exporta ONNX de vuelo.")
    return 0


def exportar(model, ckpt_path: str, onnx_out: str, size: int) -> None:
    """Exporta EL MEJOR checkpoint (dynamo=False: requisito IMX500/cv2.dnn)."""
    from cansat.checkpoints import load_model_state
    model.load_state_dict(load_model_state(ckpt_path, strict=True))
    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, size, size), onnx_out,
                      input_names=["input"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"[OK] {onnx_out} (modelo BEST, dynamo off)")
    print(f"     Verificá: python audit_imx500.py --onnx {onnx_out}")


if __name__ == "__main__":
    raise SystemExit(main())
