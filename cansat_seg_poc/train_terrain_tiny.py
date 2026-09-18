"""
CanSat La Base — F1.2: modelo de terreno "tiny" para Pi Zero W v1 / IMX500.

Arquitecturas (todas ONNX opset 17, una entrada, salida 'logits'):

  lraspp-mv3s   MobileNetV3-Small + LR-ASPP (~1.6M params).
                Prioridad 1 del plan: solo convs/depthwise/SE, sin attention.
  deeplabv3plus DeepLabV3+ MobileNetV2 (control: aísla el efecto del backbone).

Reusa dataset, pesos de clase y métricas de ``train_terrain_v2.py``.

Uso:
    python train_terrain_tiny.py --arch lraspp-mv3s --size 224 --epochs 30
    python train_terrain_tiny.py --arch deeplabv3plus --size 224 --epochs 30 \
        --out outputs/best_terrain_v2_224.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cansat.checkpoints import save_ckpt
from cansat.seed import set_seed
from train_terrain_v2 import (CLASSES, LoveDAv2, compute_weights,
                              conf_matrix, find_pairs)


class LRASPPMobileNetV3Small(nn.Module):
    """
    MobileNetV3-Small (ImageNet) + LR-ASPP.

    ``low``  : features[:4]  → stride 8, 24 canales (detalle).
    ``high`` : features[4:12] → stride 32, 576 canales (contexto).
    Cabeza LR-ASPP: rama 1x1 + compuerta global (Sigmoid), suma con la rama
    de baja resolución y upsample a la entrada.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = False,
                 mid_ch: int = 128):
        super().__init__()
        from torchvision.models import (MobileNet_V3_Small_Weights,
                                        mobilenet_v3_small)

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        feats = mobilenet_v3_small(weights=weights).features
        blocks = list(feats.children())             # 13 bloques (0..12)
        self.low = nn.Sequential(*blocks[:4])       # /8, 24 ch
        self.high = nn.Sequential(*blocks[4:13])    # /32, 576 ch (incluye el conv final)

        self.cbr = nn.Sequential(
            nn.Conv2d(576, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(576, mid_ch, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(24, num_classes, 1)
        self.high_classifier = nn.Conv2d(mid_ch, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        low = self.low(x)
        high = self.high(low)
        h = self.cbr(high)
        h = h * self.scale(high)
        h = self.high_classifier(h)
        h = F.interpolate(h, size=low.shape[-2:], mode="bilinear",
                          align_corners=False)
        out = self.low_classifier(low) + h
        return F.interpolate(out, size=size, mode="bilinear",
                             align_corners=False)


def build(arch: str, num_classes: int, pretrained: bool):
    if arch == "lraspp-mv3s":
        return LRASPPMobileNetV3Small(num_classes, pretrained=pretrained)
    if arch == "deeplabv3plus":
        from train import DeepLabV3PlusMobileNetV2
        return DeepLabV3PlusMobileNetV2(num_classes)
    raise ValueError(f"arquitectura desconocida: {arch}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Entrena el modelo de terreno tiny (F1) a 224/192 px")
    ap.add_argument("--arch", choices=("lraspp-mv3s", "deeplabv3plus"),
                    default="lraspp-mv3s")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/best_terrain_tiny.pth")
    ap.add_argument("--workers", type=int, default=0,
                    help="workers del DataLoader (0 = carga en el hilo principal; "
                         "en Windows usar >0 acelera pero requiere el guard de main)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="sin pesos ImageNet (para exportar sin red)")
    args = ap.parse_args()

    set_seed(args.seed)

    train_pairs = find_pairs("Train")
    val_pairs = find_pairs("Val")
    if not train_pairs:
        print("[ERROR] No encontré pares en dataset/loveda_remapped/Train")
        return 1
    print(f"Pares: train {len(train_pairs)} / val {len(val_pairs)} · "
          f"arch={args.arch} · {args.size}px")

    w = compute_weights(train_pairs)
    print(f"Pesos por clase: {np.round(w, 2).tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build(args.arch, 5, pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {n_params:,} ({n_params * 4 / 1e6:.1f} MB FP32)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_dl = DataLoader(LoveDAv2(train_pairs, args.size, aug=True),
                          batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, persistent_workers=args.workers > 0)
    val_dl = DataLoader(LoveDAv2(val_pairs, args.size),
                        batch_size=args.batch,
                        num_workers=args.workers, persistent_workers=args.workers > 0)

    w_t = torch.tensor(w).to(device)
    best = -1.0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for bi, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=w_t, ignore_index=255)
            loss.backward()
            opt.step()
            run += loss.item()
            if bi % 50 == 0:
                print(f"    batch {bi}/{len(train_dl)} loss {loss.item():.4f}",
                      flush=True)
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
        print(f"  epoch {ep + 1}: mIoU {miou:.3f} | loss {run / len(train_dl):.4f}"
              + "  ".join(f" | {CLASSES[i][:4]}={ious[i]:.2f}" for i in range(5)),
              flush=True)
        if miou > best:
            best = miou
            save_ckpt(args.out, model, num_classes=5, img_size=args.size,
                      class_names=list(CLASSES),
                      miou=round(miou, 4),
                      iou_per_class=[round(float(x), 4) for x in ious],
                      dataset="loveda_remapped", script="train_terrain_tiny.py",
                      arch=args.arch, epochs=ep + 1, seed=args.seed)
            print(f"  → guardado {args.out}", flush=True)

    print(f"[OK] mejor mIoU {args.arch}@{args.size}: {best:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
