"""
CanSat La Base — Eval: sliding window + TTA vs resize completo.
Compara IoU de "dañado" con 3 estrategias de inferencia sobre xBD val.

⚠ CORREGIDO: la versión anterior barajaba TODO el manifest (que apunta a
  ``xbd/train``) y evaluaba los primeros 20 tiles. Medido: 18 de 20 estaban en
  el train de ``train_damage*.py`` (split aleatorio por fila), así que el "IoU
  de val" era en realidad IoU sobre datos vistos, y además mezclaba tiles del
  mismo desastre entre train y val.

Ahora el split es **por desastre** (agrupando por el prefijo del nombre, p. ej.
``hurricane-harvey_*``): los grupos de ``--split val`` nunca aparecen en el
train. Es la única forma de que el número sea citable.

Uso:
    python eval_sliding.py --ckpt outputs/best_damage3.pth --tiles 20
    python eval_sliding.py --split train --tiles 20      # comparar con train
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from cansat.xbd import split_por_desastre
from train import DeepLabV3PlusMobileNetV2

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MASKS = Path("dataset/xbd_masks")


def prep(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).float()[None]


def positions(total: int, win: int, stride: int) -> list[int]:
    """
    Posiciones de inicio de ventana, clampeadas a ``[0, max(total-win, 0)]``.

    ⚠ ARREGLADO: antes era ``range(0, h-size+1, stride) + [h-size]``. Con
    ``h < size`` el último término es **negativo**, el tile sale con índice
    negativo y el acumulado queda desalineado.
    """
    last = max(int(total) - int(win), 0)
    pos = list(range(0, last + 1, max(1, int(stride))))
    if not pos or pos[-1] != last:
        pos.append(last)
    return sorted({max(0, min(p, last)) for p in pos})


def infer_tile(model, tile, size, device):
    """Logits (3, th, tw); rellena con espejo si el tile es menor que la ventana."""
    th, tw = tile.shape[:2]
    if th < size or tw < size:
        tile = cv2.copyMakeBorder(tile, 0, max(0, size - th), 0, max(0, size - tw),
                                  cv2.BORDER_REFLECT_101)
    lg = model(prep(tile).to(device))[0].cpu().numpy()
    return lg[:, :th, :tw]


def iou_dano(pred, gt):
    """Daño binario: pred == 2 (clase dañado) vs gt >= 2 (minor+major+destroyed)."""
    p, g = pred == 2, gt >= 2
    u = (p | g).sum()
    return (p & g).sum() / u if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/best_damage3.pth")
    ap.add_argument("--tiles", type=int, default=20)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--stride", type=int, default=160)
    ap.add_argument("--split", choices=("val", "train", "todos"), default="val",
                    help="val = desastres reservados (nunca vistos en train)")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3PlusMobileNetV2(3).to(device)
    # load_model_state acepta los DOS formatos de checkpoint (dict con metadata o
    # state_dict crudo). Antes asumia crudo y reventaba con los nuevos.
    from cansat.checkpoints import load_model_state
    model.load_state_dict(load_model_state(args.ckpt))
    model.eval()

    with open(MASKS / "manifest.csv", encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh))
    train_rows, val_rows, val_grupos = split_por_desastre(
        manifest, args.holdout_frac, args.seed)
    if args.split == "val":
        rows = val_rows
        print(f"Split por desastre: val = {val_grupos}")
        print(f"  ({len(val_rows)} tiles de {len(val_grupos)} desastres nunca "
              f"vistos; train tiene {len(train_rows)})")
    elif args.split == "train":
        rows = train_rows
        print(f"Split por desastre: train ({len(train_rows)} tiles)")
    else:
        rows = manifest
        print("Sin split (todos los tiles: número NO citable, hay train adentro)")

    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.tiles]

    res = {"full": [], "slide": [], "slide_tta": []}
    with torch.no_grad():
        for r in rows:
            img = cv2.imread(r["image"])
            gt = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
            if img is None or gt is None:
                continue
            h, w = img.shape[:2]

            # 1) full resize (método actual)
            lg = model(prep(cv2.resize(img, (args.size, args.size))).to(device))
            full = cv2.resize(lg[0].argmax(0).cpu().numpy().astype(np.uint8),
                              (w, h), interpolation=cv2.INTER_NEAREST)

            # 2) sliding window (+ TTA de 4 vistas, igual que el pipeline)
            for key, tta in [("slide", False), ("slide_tta", True)]:
                acc = np.zeros((3, h, w), np.float32)
                cnt = np.zeros((h, w), np.float32)
                for y in positions(h, args.size, args.stride):
                    for x in positions(w, args.size, args.stride):
                        th = min(args.size, h - y)
                        tw = min(args.size, w - x)
                        tile = img[y:y + th, x:x + tw]
                        l1 = infer_tile(model, tile, args.size, device)
                        acc[:, y:y + th, x:x + tw] += l1
                        cnt[y:y + th, x:x + tw] += 1
                        if tta:
                            for flip, _eje in ((1, 1), (0, 0), (-1, None)):
                                t2 = cv2.flip(tile, flip)
                                lf = infer_tile(model, t2, args.size, device)
                                if flip == 1:
                                    lf = lf[:, :, ::-1]
                                elif flip == 0:
                                    lf = lf[:, ::-1, :]
                                else:
                                    lf = lf[:, ::-1, ::-1]
                                acc[:, y:y + th, x:x + tw] += lf
                                cnt[y:y + th, x:x + tw] += 1
                pred = (acc / np.maximum(cnt, 1e-6)).argmax(0)
                res[key].append(iou_dano(pred, gt))
            res["full"].append(iou_dano(full, gt))

    print(f"\nIoU dañado sobre {len(rows)} tiles "
          f"(split={args.split}, semilla {args.seed}):")
    for k, vals in res.items():
        print(f"  {k:10s}: {np.mean(vals):.3f}  (mediana {np.median(vals):.3f}, "
              f"n={len(vals)})")


if __name__ == "__main__":
    main()
