"""
CanSat La Base — Fase 4a: mejor modelo de terreno (LoveDA v2).
Multi-escala + class-weights + augmentación de vuelo + cosine LR.

Uso:
    python train_terrain_v2.py --epochs 12
"""
import argparse
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
CLASSES = ["vegetation", "building", "water", "bare_ground", "other"]
ROOT = Path("dataset/loveda_remapped")


def degrade_flight(img):
    """Borrón de movimiento + ruido leve (lo que aprendimos del bench v2)."""
    # np.random.default_rng() SIN argumento ignora np.random.seed: cada corrida
    # era distinta aunque se fijara la semilla. Con el generador global
    # sembrado (set_seed) la augmentación es reproducible.
    h, w = img.shape[:2]
    if random.random() < 0.5:
        img = cv2.resize(cv2.resize(img, (w // 2, h // 2)), (w, h))
    k = 9
    kern = np.zeros((k, k))
    kern[k // 2, :] = 1.0 / k
    img = cv2.filter2D(img, -1, kern)
    return np.clip(img.astype(np.float32) + np.random.normal(0, 4, img.shape),
                   0, 255).astype(np.uint8)


def jitter(img):
    a = random.uniform(0.8, 1.2)
    b = random.uniform(-20, 20)
    return np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


class LoveDAv2(Dataset):
    def __init__(self, pairs, size=320, aug=False):
        self.pairs, self.size, self.aug = pairs, size, aug

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.imread(str(ip))
        msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if self.aug and random.random() < 0.5 and min(img.shape[:2]) > 512:
            h, w = img.shape[:2]
            y0 = random.randint(0, h - 512)
            x0 = random.randint(0, w - 512)
            img = img[y0:y0 + 512, x0:x0 + 512]
            msk = msk[y0:y0 + 512, x0:x0 + 512]
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
            if random.random() < 0.3:
                img = degrade_flight(img)
            if random.random() < 0.5:
                img = jitter(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        msk[msk >= 5] = 255  # clase "ignore" de LoveDA
        return (torch.from_numpy(rgb.transpose(2, 0, 1)).float(),
                torch.from_numpy(msk.astype(np.int64)))


def find_pairs(split):
    pairs = []
    for sub in ("Urban", "Rural"):
        ib = ROOT / split / sub / "images_png"
        mb = ROOT / split / sub / "masks_png"
        if not ib.exists():
            continue
        pairs += [(ip, mb / ip.name) for ip in sorted(ib.glob("*.png"))
                  if (mb / ip.name).exists()]
    return pairs


def compute_weights(pairs):
    tot = np.zeros(5)
    for _, mp in pairs:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
        tot += np.bincount(m.ravel(), minlength=5)[:5]
    w = 1.0 / np.sqrt(tot + 1)
    w = w / w.mean()
    return np.clip(w, 0.3, 8).astype(np.float32)


def conf_matrix(pred, targ, n=5):
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/best_terrain_v2.pth")
    args = ap.parse_args()

    # Semilla completa (random+numpy+torch+cudnn): antes no había NINGUNA y
    # `degrade_flight` usaba np.random sin sembrar → cada corrida era distinta.
    from cansat.seed import set_seed
    set_seed(args.seed)

    train_pairs = find_pairs("Train")
    val_pairs = find_pairs("Val")
    if not train_pairs:
        print(f"[ERROR] No encontré pares en {ROOT}/train/images+masks")
        raise SystemExit(1)
    print(f"Pares: train {len(train_pairs)} / val {len(val_pairs)}")

    w = torch.tensor(compute_weights(train_pairs))
    print(f"Pesos por clase: {np.round(w.numpy(), 2).tolist()}")

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
        state = model.state_dict()
        filtered = {k: v for k, v in sd.items()
                    if k in state and state[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        print(f"Checkpoint v1 cargado ({len(filtered)}/{len(state)} capas)")
    model = model.to(device)
    w = w.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_dl = DataLoader(LoveDAv2(train_pairs, args.size, aug=True),
                          batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(LoveDAv2(val_pairs, args.size), batch_size=args.batch)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        for bi, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=w, ignore_index=255)
            loss.backward()
            opt.step()
            if bi % 50 == 0:
                print(f"    batch {bi}/{len(train_dl)} loss {loss.item():.4f}")
        sched.step()

        model.eval()
        cm = np.zeros((5, 5), dtype=np.int64)
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x.to(device)).argmax(1).cpu().numpy()
                cm += conf_matrix(pred, y.numpy())
        ious = []
        for i in range(5):
            inter = cm[i, i]
            union = cm[i, :].sum() + cm[:, i].sum() - inter
            ious.append(inter / union if union else 0.0)
        miou = float(np.mean(ious))
        print(f"  epoch {ep + 1}: mIoU {miou:.3f} | "
              + "  ".join(f"{CLASSES[i][:4]}={ious[i]:.2f}" for i in range(5)))
        if miou > best:
            best = miou
            # Formato canónico (dict + metadata) en vez de state_dict crudo:
            # así el exportador y el registro saben de dónde salió cada peso.
            from cansat.checkpoints import save_ckpt
            save_ckpt(args.out, model, num_classes=5, img_size=args.size,
                      class_names=list(CLASSES),
                      miou=round(float(miou), 4),
                      iou_per_class=[round(float(x), 4) for x in ious],
                      dataset="loveda_remapped", script="train_terrain_v2.py",
                      epochs=ep + 1, seed=args.seed)
            print(f"  → guardado {args.out}")

    print(f"[OK] mejor mIoU terreno v2: {best:.3f}")

    # ⚠ Se exporta EL MEJOR checkpoint, no el modelo en memoria de la última
    #   época (antes el ONNX no correspondía al .pth guardado), y con
    #   dynamo=False: el default de torch 2.11 genera IR10 + pesos externos,
    #   que ni cv2.dnn (Pi) ni el conversor IMX500 aceptan.
    from cansat.checkpoints import load_model_state
    model.load_state_dict(load_model_state(args.out, strict=True))
    model.eval().cpu()
    dummy = torch.randn(1, 3, args.size, args.size)
    torch.onnx.export(model, dummy,
                      "outputs/cansat_seg_terrain_v2.onnx",
                      input_names=["input"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True,
                      dynamo=False)
    print("[OK] outputs/cansat_seg_terrain_v2.onnx (BEST, dynamo off)")

if __name__ == "__main__":
    main()
