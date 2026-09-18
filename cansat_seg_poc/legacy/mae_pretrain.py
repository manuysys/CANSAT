"""
CanSat La Base — MAE (masked autoencoding) sobre MobileNetV2, autocontenido.
Uso: python mae_pretrain.py --data images --epochs 100 --batch 32
"""

# ── legacy/: script fuera del pipeline activo. Para correrlo desde la raíz:
#     python legacy/mae_pretrain.py
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v2

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class ImgFolder(Dataset):
    def __init__(self, root, size=224):
        self.files = sorted([p for p in Path(root).rglob("*")
                             if p.suffix.lower() in (".jpg", ".png", ".jpeg")
                             and not any("mask" in q.lower() for q in p.parts)])
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(str(self.files[i]))
        img = cv2.resize(img, (self.size, self.size))
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        if random.random() < 0.5:
            img = cv2.flip(img, 0)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1))


def make_mask(size=224, patch=16, ratio=0.6):
    g = size // patch
    m = torch.zeros(g * g)
    m[torch.randperm(g * g)[:int(g * g * ratio)]] = 1.0
    m = m.view(g, g)
    return m.repeat_interleave(patch, 0).repeat_interleave(patch, 1)


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = mobilenet_v2(pretrained=True).features
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, 2, 2), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 2, 2), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 2, 2), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 2, 2))

    def forward(self, x, mask):
        xm = x * (1 - mask)[None, None]
        rec = self.dec(self.enc(xm))
        tgt = F.interpolate(x, size=112, mode="bilinear", align_corners=False)
        m112 = F.interpolate(mask[None, None], size=112)[0, 0] > 0.5
        return ((rec - tgt) ** 2 * m112[None, None]).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="images")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # Semilla completa (random+numpy+torch+cudnn): este script no fijaba
    # NINGUNA y la augmentacion usa el modulo random global.
    from cansat.seed import set_seed
    set_seed(getattr(args, "seed", 42))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl = DataLoader(ImgFolder(args.data), batch_size=args.batch, shuffle=True,
                    num_workers=4, pin_memory=True)
    model = MAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for x in dl:
            x = x.to(device)
            mask = make_mask().to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = model(x, mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            n += 1
        torch.save(model.enc.state_dict(), "mae_mobilenetv2_encoder.pth")
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1}: loss {tot/n:.4f}")
    print("[OK] mae_mobilenetv2_encoder.pth")


if __name__ == "__main__":
    main()
