"""
CanSat La Base — v3: fine-tune semi-supervisado (real + pseudo-labels)
con backbone inicializado desde el MAE aéreo.

Uso:
    python train_v3.py --epochs 8 --batch-size 8
    python train_v3.py --backbone cbam ...   (variante de control)
"""

# ── legacy/: script fuera del pipeline activo. Para correrlo desde la raíz:
#     python legacy/train_v3.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train import (LoveDADataset, confusion_of, metrics_from_conf,
                   class_weights, CLASS_NAMES, IGNORE_INDEX)
from train_cbam import DeepLabV3PlusCBAM

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class PseudoDataset(Dataset):
    def __init__(self, manifest, size=320, augment=True, skip=""):
        self.items = [it for it in json.load(open(manifest, encoding="utf-8"))
                      if skip not in it["image"].lower()]
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.imread(f"dataset/pseudo/img320/{Path(it['mask']).stem}.jpg")
        msk = cv2.imread(it["mask"], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.size, self.size))
        msk = cv2.resize(msk, (self.size, self.size),
                         interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if random.random() < 0.5:
                img, msk = cv2.flip(img, 1), cv2.flip(msk, 1)
            if random.random() < 0.5:
                img, msk = cv2.flip(img, 0), cv2.flip(msk, 0)
            k = random.choice([0, 1, 2, 3])
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))
                msk = np.ascontiguousarray(np.rot90(msk, k))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return (torch.from_numpy(x.transpose(2, 0, 1)),
                torch.from_numpy(msk.astype(np.int64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="dataset/loveda_remapped")
    ap.add_argument("--manifest", default="dataset/pseudo/manifest.json")
    ap.add_argument("--backbone", choices=["mae", "cbam"], default="mae")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pseudo-weight", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)          # faltaba: la augmentación numpy no era reproducible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepLabV3PlusCBAM(5).to(device)
    sd = torch.load("outputs/best_terrain_cbam.pth", map_location="cpu")
    model.load_state_dict(sd, strict=False)
    if args.backbone == "mae":
        mae = torch.load("mae_mobilenetv2_encoder.pth", map_location="cpu")
        model.backbone.load_state_dict(mae, strict=False)
        print(f"[OK] init: head CBAM + backbone MAE ({len(mae)} claves)")
        lr_b, lr_h = 1e-4, 2e-5
    else:
        print("[OK] init: CBAM completo (control)")
        lr_b, lr_h = 3e-5, 2e-5

    real_ds = LoveDADataset(args.dataset_root, "Train", 320, augment=True)
    val_ds = LoveDADataset(args.dataset_root, "Val", 320, augment=False)
    pseudo_ds = PseudoDataset(args.manifest, 320, augment=True, skip="floodnet")
    print(f"[OK] real {len(real_ds)} + pseudo {len(pseudo_ds)}")
    real_dl = DataLoader(real_ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, pin_memory=True)
    pseudo_dl = DataLoader(pseudo_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=0, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    weight = class_weights(args.dataset_root).to(device)
    crit = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr_b},
        {"params": [p for n, p in model.named_parameters()
                    if not n.startswith("backbone")], "lr": lr_h},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()   # no contaminar running stats con pseudo
        it_p = iter(pseudo_dl)
        for bi, (x, y) in enumerate(real_dl):          # ← cambia esta línea
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
                try:
                    xp, yp = next(it_p)
                except StopIteration:
                    it_p = iter(pseudo_dl)
                    xp, yp = next(it_p)
                xp, yp = xp.to(device), yp.to(device)
                loss = loss + args.pseudo_weight * crit(model(xp), yp)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if bi % 20 == 0:                            # ← agregá estas 2 líneas
                print(f"    batch {bi}/{len(real_dl)} loss {loss.item():.3f}",
                      flush=True)
        sched.step()

        conf = confusion_of(model, val_dl, device)
        miou, iou, f1 = metrics_from_conf(conf)
        tag = ""
        if miou > best:
            best = miou
            from cansat.checkpoints import save_ckpt
            save_ckpt("outputs/best_terrain_v3.pth", model,
                      num_classes=5, img_size=320, class_names=list(CLASS_NAMES),
                      miou=round(float(miou), 4),
                      iou_per_class=[round(float(x), 4) for x in iou],
                      dataset="loveda_remapped + pseudo", backbone=args.backbone,
                      script="train_v3.py", epochs=ep + 1)
            tag = " → guardado"
        print(f"  epoch {ep+1}: mIoU {miou:.4f} | "
              + "  ".join(f"{CLASS_NAMES[i][:4]}={iou[i]:.2f}" for i in range(5))
              + tag)

    print(f"[OK] mejor mIoU v3 ({args.backbone}): {best:.4f}")
    # ⚠ Se exporta EL MEJOR checkpoint (antes se exportaba la última época) y el
    #   nombre de salida coincide con el que se imprime: era
    #   `cansat_seg_terrain_v4.onnx` con un print que decía v3.
    from cansat.checkpoints import load_model_state
    model.load_state_dict(load_model_state("outputs/best_terrain_v3.pth", strict=True))
    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, 320, 320),
                      "outputs/cansat_seg_terrain_v3.onnx",
                      input_names=["input"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True,
                      dynamo=False)
    print("[OK] outputs/cansat_seg_terrain_v3.onnx (BEST, dynamo off)")


if __name__ == "__main__":
    main()
