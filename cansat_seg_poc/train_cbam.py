"""
CanSat La Base — CBAM sobre la arquitectura real (DeepLabV3+ de train.py),
inicializado 100% desde el destilado. CBAM arranca como identidad:
el modelo NUNCA empeora el baseline; solo aprende si hay ganancia.

Uso:
    python train_cbam.py --epochs 10 --batch-size 8
"""
import argparse
import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import (LoveDADataset, DeepLabV3PlusMobileNetV2, confusion_of,
                   metrics_from_conf, class_weights, CLASS_NAMES,
                   IGNORE_INDEX)


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(ch, ch // r, 1, bias=False),
                                 nn.ReLU(True),
                                 nn.Conv2d(ch // r, ch, 1))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        a = self.mlp(F.adaptive_avg_pool2d(x, 1))
        m = self.mlp(F.adaptive_max_pool2d(x, 1))
        return 2.0 * torch.sigmoid(a + m)          # init = 1 → identidad


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        a = x.mean(1, keepdim=True)
        m, _ = x.max(1, keepdim=True)
        return 2.0 * torch.sigmoid(self.conv(torch.cat([a, m], 1)))


class CBAM(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.ca = ChannelAttention(ch, r)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        return x * self.sa(x)


class DeepLabV3PlusCBAM(DeepLabV3PlusMobileNetV2):
    """Tu arquitectura exacta + CBAM post-ASPP y post-decoder."""
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.cbam_aspp = CBAM(256)
        self.cbam_dec = CBAM(256)

    def forward(self, x):
        size = x.shape[-2:]
        low = None
        for idx, block in enumerate(self.backbone):
            x = block(x)
            if idx == 3:
                low = x
        x = self.cbam_aspp(self.aspp(x))
        x = F.interpolate(x, size=low.shape[-2:],
                          mode="bilinear", align_corners=False)
        x = torch.cat([x, self.low_conv(low)], dim=1)
        h = self.decoder[:4](x)          # conv3x3+BN+ReLU+dropout
        h = self.cbam_dec(h)
        h = self.decoder[4](h)           # clasificador 1x1
        return F.interpolate(h, size=size, mode="bilinear", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=os.path.join("dataset", "loveda_remapped"))
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--init", default="outputs/best_terrain_v2_distilled.pth")
    ap.add_argument("--workers", type=int, default=0,
                    help="workers del DataLoader (0 = seguro en Windows/Pi)")
    args = ap.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)          # faltaba: augmentación numpy no reproducible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model = DeepLabV3PlusCBAM(5).to(device)
    # load_into acepta los dos formatos y ABORTA si carga menos de la mitad de
    # las capas (antes un load_state_dict(strict=False) podía cargar 0 en
    # silencio si el checkpoint venía en el formato dict nuevo).
    from cansat.checkpoints import load_into
    _, frac = load_into(model, args.init, strict=False, min_loaded_frac=0.5,
                        verbose=True)
    print(f"[OK] cargado {args.init} ({frac:.0%} de las capas)")

    train_ds = LoveDADataset(args.dataset_root, "Train", args.img_size, augment=True)
    val_ds = LoveDADataset(args.dataset_root, "Val", args.img_size, augment=False)
    # num_workers=0 por defecto: 4 hardcodeado rompe en Windows (spawn) y en la
    # Pi; se puede subir con --workers.
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        num_workers=args.workers)

    weight = class_weights(args.dataset_root).to(device)
    crit = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)

    # CBAM rápido, resto muy suave (para no romper lo aprendido)
    opt = torch.optim.AdamW([
        {"params": list(model.cbam_aspp.parameters()) + list(model.cbam_dec.parameters()),
         "lr": 1e-3},
        {"params": list(model.backbone.parameters()) + list(model.aspp.parameters())
                   + list(model.low_conv.parameters()) + list(model.decoder.parameters()),
         "lr": 3e-5},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # AMP sólo con CUDA: GradScaler("cuda") incondicional revienta en CPU
    # (mismo patrón que train.py, que usa enabled=).
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for xb, yb in train_dl:
            x, y = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        conf = confusion_of(model, val_dl, device)
        miou, iou, _f1 = metrics_from_conf(conf)
        tag = ""
        if miou > best:
            best = miou
            from cansat.checkpoints import save_ckpt
            save_ckpt("outputs/best_terrain_cbam.pth", model,
                      num_classes=5, img_size=args.img_size,
                      class_names=list(CLASS_NAMES),
                      miou=round(float(miou), 4),
                      iou_per_class=[round(float(x), 4) for x in iou],
                      dataset="loveda_remapped", script="train_cbam.py",
                      epochs=ep + 1)
            tag = " → guardado"
        print(f"  epoch {ep+1}: mIoU {miou:.4f} | "
              + "  ".join(f"{CLASS_NAMES[i][:4]}={iou[i]:.2f}" for i in range(5))
              + tag)

    print(f"[OK] mejor mIoU CBAM: {best:.4f}")

    # Se exporta EL MEJOR checkpoint (antes: última época) y con dynamo=False.
    from cansat.checkpoints import load_model_state
    model.load_state_dict(load_model_state("outputs/best_terrain_cbam.pth", strict=True))
    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, args.img_size, args.img_size),
                      "outputs/cansat_seg_terrain_v2_cbam.onnx",
                      input_names=["input"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True,
                      dynamo=False)
    print("[OK] outputs/cansat_seg_terrain_v2_cbam.onnx (BEST, dynamo off)")


if __name__ == "__main__":
    main()
