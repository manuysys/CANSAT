"""
CanSat La Base — Fine-tune de terreno con init MAE. Autocontenido.
Uso: python mae_finetune_terrain.py --data dataset_loveda_remapped --epochs 12
"""

# ── legacy/: script fuera del pipeline activo. Para correrlo desde la raíz:
#     python legacy/mae_finetune_terrain.py
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
CLASSES = ["vege", "buil", "wate", "bare", "othe"]


class LoveDA(Dataset):
    def __init__(self, root, split, size=320, aug=False):
        split = {"train": "Train", "val": "Val"}[split]
        self.pairs = []
        for dom in ["Rural", "Urban"]:
            d = Path(root) / split / dom
            for ip in sorted((d / "images_png").glob("*.png")):
                mp = d / "masks_png" / ip.name
                if mp.exists():
                    self.pairs.append((ip, mp))
        self.size, self.aug = size, aug

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
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return (torch.from_numpy(x.transpose(2, 0, 1)),
                torch.from_numpy(msk.astype(np.int64)))


class ASPP(nn.Module):
    def __init__(self, in_ch=1280, out_ch=256):
        super().__init__()
        self.b0 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.b1 = nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False)
        self.b2 = nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False)
        self.b3 = nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(in_ch, out_ch, 1, bias=False))
        self.proj = nn.Conv2d(out_ch * 5, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        p = F.interpolate(self.pool(x), size=x.shape[-2:],
                          mode="bilinear", align_corners=False)
        c = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), p], 1)
        return F.relu(self.bn(self.proj(c)), inplace=True)


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.backbone = mobilenet_v2(pretrained=False).features
        self.aspp = ASPP()
        self.head = nn.Conv2d(256, n_cls, 1)

    def forward(self, x):
        out = self.head(self.aspp(self.backbone(x)))
        return F.interpolate(out, size=x.shape[-2:],
                             mode="bilinear", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # Semilla completa (random+numpy+torch+cudnn): este script no fijaba
    # NINGUNA y la augmentacion usa el modulo random global.
    from cansat.seed import set_seed
    set_seed(getattr(args, "seed", 42))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus(5).to(device)
    if Path("mae_mobilenetv2_encoder.pth").exists():
        model.backbone.load_state_dict(
            torch.load("mae_mobilenetv2_encoder.pth", map_location="cpu"))
        print("[OK] init desde MAE")
    else:
        model.backbone.load_state_dict(mobilenet_v2(pretrained=True).features.state_dict())
        print("[WARN] sin MAE, init ImageNet")

    train_dl = DataLoader(LoveDA(args.data, "train", aug=True),
                          batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(LoveDA(args.data, "val"), batch_size=args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = F.cross_entropy(model(x), y, ignore_index=255)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        model.eval()
        inter = np.zeros(5)
        union = np.zeros(5)
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                p = model(x).argmax(1).cpu().numpy()
                t = y.cpu().numpy()
                for c in range(5):
                    inter[c] += ((p == c) & (t == c)).sum()
                    union[c] += ((p == c) | (t == c)).sum()
        ious = inter / np.maximum(union, 1)
        miou = float(ious.mean())
        tag = ""
        if miou > best:
            best = miou
            torch.save(model.state_dict(), "best_terrain_mae.pth")
            tag = " → guardado"
        print(f"  epoch {ep+1}: mIoU {miou:.3f} | "
              + "  ".join(f"{CLASSES[i]}={ious[i]:.2f}" for i in range(5)) + tag)

    print(f"[OK] mejor mIoU MAE-finetune: {best:.3f} (vuelo actual: 0.529)")
    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, 320, 320), "terrain_mae_320.onnx",
                      input_names=["input"], output_names=["logits"],
                      opset_version=17)
    print("[OK] terrain_mae_320.onnx")


if __name__ == "__main__":
    main()
