"""
CanSat La Base — Fase 4c: daños two-stage (receta xView2 + loss RescueNet).
Patch-mining + crops nativos + loss de localización (BCE edificio/fondo)
combinada con la CE selectiva en foreground.

⚠ CAMBIOS 2026-09-17:
  · Split POR DESASTRE (antes por fila → fuga geográfica).
  · `--loss rescue`: adaptación de la loss de RescueNet (Rahnemoonfar et al.,
    2020) a un modelo de una sola cabeza. El paper entrena una red multi-head
    con BCE de segmentación de edificios + CE selectiva sólo en foreground y
    reporta una mejora de un orden de magnitud en las clases intermedias
    frente a la CE sola. Acá se reconstruye esa señal desde los logits de 3
    clases:  p(edificio) = 1 - p(other)  →  BCE contra (y >= 1), más la CE
    restringida a edificios (ignore_index=0).
  · Selección del mejor checkpoint por IoU de daño sobre edificios (la métrica
    de la misión), con el split honesto.

Uso:
    python train_damage_v3.py --epochs 12 --loss rescue
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
from cansat.xbd import split_por_desastre
from train import DeepLabV3PlusMobileNetV2

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MANIFEST = Path("dataset/xbd_masks/manifest.csv")


def rescue_loss(logits: torch.Tensor, y: torch.Tensor,
                weight: torch.Tensor, lambda_loc: float = 1.0) -> torch.Tensor:
    """
    Loss de localización + clasificación selectiva (adaptación de RescueNet).

    ``logits``: (B, 3, H, W) — 0=other, 1=intacto, 2=dañado.
    ``y``:      (B, H, W) con las mismas clases.

    · Término de localización: BCE entre ``p(edificio) = 1 - p(other)`` y el
      target binario ``(y >= 1)``. Enseña a separar construcción de fondo
      aunque la CE selectiva no mire el fondo.
    · Término de clasificación: CE con pesos, ``ignore_index=0`` (sólo
      edificios aportan gradiente), como ya hacía la versión anterior.
    """
    ce = F.cross_entropy(logits, y, weight=weight, ignore_index=0)
    p_fg = 1.0 - F.softmax(logits, dim=1)[:, 0]              # (B, H, W)
    t_fg = (y >= 1).float()
    bce = F.binary_cross_entropy(p_fg.clamp(1e-6, 1 - 1e-6), t_fg)
    return ce + lambda_loc * bce


class XBDv3(Dataset):
    def __init__(self, rows, size=320, aug=False):
        self.rows, self.size, self.aug = rows, size, aug

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(r["image"])
        msk = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
        msk = np.where(msk >= 2, 2, msk).astype(np.uint8)
        h, w = img.shape[:2]
        s = self.size
        if self.aug:
            ys, xs = np.nonzero(msk >= 1)          # patch-mining: edificios
            if len(ys) and random.random() < 0.8:
                j = random.randrange(len(ys))
                y0 = int(min(max(ys[j] - s // 2, 0), h - s))
                x0 = int(min(max(xs[j] - s // 2, 0), w - s))
                img = img[y0:y0 + s, x0:x0 + s]
                msk = msk[y0:y0 + s, x0:x0 + s]
        if img.shape[0] != s or img.shape[1] != s:
            img = cv2.resize(img, (s, s))
            msk = cv2.resize(msk, (s, s), interpolation=cv2.INTER_NEAREST)
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


def conf_matrix(pred, targ, n=3):
    pred, targ = pred.ravel(), targ.ravel()
    ok = (targ >= 0) & (targ < n)
    return np.bincount(n * targ[ok].astype(int) + pred[ok].astype(int),
                       minlength=n * n).reshape(n, n)


def leer_manifest(path: str) -> list[dict]:
    """Filas de un manifest con el formato de xBD (name,image,mask,intacto,...)."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def iou_dano_loader(model, loader, device) -> tuple[float, np.ndarray]:
    """IoU de dañado two-stage + matriz de confusión sobre un loader."""
    cm = np.zeros((3, 3), dtype=np.int64)
    inter_d = union_d = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            yt = y.numpy()
            cm += conf_matrix(pred, yt)
            pd, gd, gb = pred == 2, yt == 2, yt >= 1
            inter_d += int((pd & gd).sum())          # FP en fondo no cuentan
            union_d += int(((pd & gb) | gd).sum())   # = two-stage real
    return (inter_d / union_d if union_d else 0.0), cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="outputs/best_damage_v3.pth")
    ap.add_argument("--loss", choices=("ce", "rescue"), default="rescue",
                    help="ce = sólo CE selectiva (comportamiento viejo) | "
                         "rescue = + BCE de localización edificio/fondo")
    ap.add_argument("--lambda-loc", type=float, default=1.0,
                    help="peso del término de localización de RescueNet")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--extra-manifest", nargs="*", default=[],
                    help="manifests adicionales (formato xBD) que van TODOS a "
                         "train; p.ej. dataset/rescuenet_tiles/manifest_train.csv")
    ap.add_argument("--extra-val-manifest", nargs="*", default=[],
                    help="manifests adicionales solo para REPORTAR validación "
                         "(no seleccionan checkpoint)")
    ap.add_argument("--xbd-repeat", type=int, default=1,
                    help="repite las filas xBD de train N veces para balancear "
                         "dominios cuando se agregan extras (p.ej. 4)")
    ap.add_argument("--workers", type=int, default=0,
                    help="workers del DataLoader (0 = hilo principal)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    rows = leer_manifest(MANIFEST)
    # ⚠ Split POR DESASTRE (ver cansat/xbd.py): el holdout son eventos enteros.
    train_rows, val_rows, val_grupos = split_por_desastre(
        rows, args.holdout_frac, args.seed)
    n_xbd = len(train_rows)
    if args.xbd_repeat > 1:
        train_rows = train_rows * args.xbd_repeat
        print(f"xBD train repetido ×{args.xbd_repeat} → {len(train_rows)} filas")
    for m in args.extra_manifest:
        extra = leer_manifest(m)
        print(f"Extra train: {len(extra)} filas de {m}")
        train_rows = train_rows + extra
    extra_val_rows = []
    for m in args.extra_val_manifest:
        extra = leer_manifest(m)
        print(f"Extra val (reporte): {len(extra)} filas de {m}")
        extra_val_rows = extra_val_rows + extra
    print(f"Muestras: xBD {len(rows)} → train {n_xbd} / val {len(val_rows)} · "
          f"train total {len(train_rows)}")
    print(f"Val por desastre: {val_grupos}")

    ti = sum(int(r["intacto"]) for r in train_rows)
    td = sum(int(r["menor"]) + int(r["mayor"]) + int(r["destruido"])
             for r in train_rows)
    w = torch.tensor([0.0, 1.0, float(np.clip(ti / max(1, td), 0.5, 5))])
    print(f"Pesos (intacto=1): {np.round(w.numpy(), 2).tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    try:
        model = DeepLabV3PlusMobileNetV2(3)
    except TypeError:
        model = DeepLabV3PlusMobileNetV2()
    ckpt = Path("outputs/best_damage3.pth")
    if ckpt.exists():
        # ⚠ load_into acepta los DOS formatos (dict con metadata o state_dict
        #   crudo) y ABORTA si carga menos capas de las exigidas. Antes esto era
        #   un `torch.load(...).items()` que asumía state_dict crudo: cuando el
        #   checkpoint pasó al formato dict, cargó 0/362 capas EN SILENCIO y el
        #   modelo entrenaba desde cero sin que nadie se enterara.
        from cansat.checkpoints import load_into
        _, frac = load_into(model, ckpt, strict=False, min_loaded_frac=0.5,
                            verbose=True)
        print(f"Checkpoint damage3 cargado ({frac:.0%} de las capas)")
    model = model.to(device)
    w = w.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_dl = DataLoader(XBDv3(train_rows, args.size, aug=True),
                          batch_size=args.batch, shuffle=True,
                          num_workers=args.workers,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(XBDv3(val_rows, args.size), batch_size=args.batch,
                        num_workers=args.workers,
                        persistent_workers=args.workers > 0)
    extra_val_dl = (DataLoader(XBDv3(extra_val_rows, args.size),
                               batch_size=args.batch,
                               num_workers=args.workers,
                               persistent_workers=args.workers > 0)
                    if extra_val_rows else None)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        for bi, (xb, yb) in enumerate(train_dl):
            x, y = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(x)
            if args.loss == "rescue":
                loss = rescue_loss(logits, y, w, args.lambda_loc)
            else:
                # loss SOLO sobre edificios (el fondo no aporta gradiente)
                loss = F.cross_entropy(logits, y, weight=w, ignore_index=0)
            loss.backward()
            opt.step()
            if bi % 50 == 0:
                print(f"    batch {bi}/{len(train_dl)} loss {loss.item():.4f}")

        model.eval()
        iou_d2, cm = iou_dano_loader(model, val_dl, device)
        ious = []
        for i in range(3):
            inter = cm[i, i]
            union = cm[i, :].sum() + cm[:, i].sum() - inter
            ious.append(inter / union if union else 0.0)
        extra_txt = ""
        if extra_val_dl is not None:
            iou_ex, cm_ex = iou_dano_loader(model, extra_val_dl, device)
            inter_e = cm_ex[2, 2]
            union_e = cm_ex[2, :].sum() + cm_ex[:, 2].sum() - inter_e
            extra_txt = (f" | extra dañado={inter_e / union_e if union_e else 0:.3f}"
                         f" two-stage={iou_ex:.3f}")
        print(f"  IoU: other={ious[0]:.2f} intacto={ious[1]:.2f} "
              f"dañado={ious[2]:.2f} | DAÑADO* xBD={iou_d2:.3f}{extra_txt}")
        if iou_d2 > best:
            best = iou_d2
            meta_extra = {}
            if extra_val_dl is not None:
                meta_extra["iou_dano_extra"] = round(float(iou_ex), 4)
            save_ckpt(args.out, model,
                      num_classes=3, img_size=args.size,
                      class_names=["other", "intacto", "danado"],
                      iou_dano_two_stage=round(iou_d2, 4),
                      iou_per_class=[round(float(x), 4) for x in ious],
                      dataset="xbd (split por desastre) + extras",
                      val_desastres=val_grupos,
                      loss=args.loss,
                      script="train_damage_v3.py",
                      epochs=ep + 1, seed=args.seed, **meta_extra)
            print(f"  → guardado {args.out}")

    print(f"[OK] mejor DAÑADO* (two-stage): {best:.3f}")


if __name__ == "__main__":
    main()
