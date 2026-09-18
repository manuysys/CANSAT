"""
CanSat La Base — SegFormer-B5 para POST-VUELO (alta calidad, sin límite de memoria).
Etapa 1: python train_segformer_b5.py --epochs 16 --size 320 --batch 2 --accum 4
Etapa 2: python train_segformer_b5.py --epochs 6 --size 512 --batch 1 --accum 8
         --lr 1e-5 --resume outputs/best_terrain_segformer_b5.pth

Dejalo corriendo de noche (~4-6 h total).
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from train_terrain_v2 import LoveDAv2, find_pairs, conf_matrix, compute_weights

CLASSES = ["vegetation", "building", "water", "bare_ground", "other"]


class Wrap(torch.nn.Module):
    def __init__(self, m, size):
        super().__init__()
        self.m, self.size = m, size

    def forward(self, x):
        return F.interpolate(self.m(x).logits, size=(self.size, self.size),
                             mode="bilinear", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/best_terrain_segformer_b5.pth")
    args = ap.parse_args()

    # Antes no había NINGUNA semilla: dos corridas idénticas daban modelos
    # distintos y el mIoU no era comparable.
    from cansat.seed import set_seed
    set_seed(args.seed)

    train_pairs = find_pairs("Train")
    val_pairs = find_pairs("Val")
    print(f"Pares: train {len(train_pairs)} / val {len(val_pairs)} "
          f"| size {args.size} | batch {args.batch}x{args.accum}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    base = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        num_labels=5, ignore_mismatched_sizes=True)
    if args.resume:
        # load_model_state acepta dict-con-metadata (save_ckpt) o state_dict crudo.
        from cansat.checkpoints import load_model_state
        base.load_state_dict(load_model_state(args.resume))
        print(f"[OK] resumed desde {args.resume}")
    model = Wrap(base, args.size).to(device)

    w = torch.tensor(compute_weights(train_pairs)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    train_dl = DataLoader(LoveDAv2(train_pairs, args.size, aug=True),
                          batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(LoveDAv2(val_pairs, args.size), batch_size=args.batch)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        for bi, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda"):
                loss = F.cross_entropy(model(x), y, weight=w,
                                       ignore_index=255) / args.accum
            scaler.scale(loss).backward()
            if (bi + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            if bi % 50 == 0:
                print(f"    batch {bi}/{len(train_dl)} "
                      f"loss {loss.item() * args.accum:.4f}")
        # ⚠ Flush de la última ventana de acumulación: si len(dl) % accum != 0,
        #   los gradientes del resto se descartaban (nunca se aplicaban).
        if len(train_dl) % args.accum != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        sched.step()

        model.eval()
        cm = np.zeros((5, 5), dtype=np.int64)
        with torch.no_grad():
            for x, y in val_dl:
                with torch.amp.autocast("cuda"):
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
            from cansat.checkpoints import save_ckpt
            # ⚠ Se guarda el state_dict de `base` (modelo HF), no el del Wrap:
            #   el Wrap es sólo para exportar (interpola los logits) y tiene las
            #   claves con prefijo, así que los exportadores cargan sobre `base`.
            save_ckpt(args.out, base,
                      num_classes=5, img_size=args.size, class_names=list(CLASSES),
                      miou=round(float(miou), 4),
                      iou_per_class=[round(float(x), 4) for x in ious],
                      dataset="loveda_remapped", script="train_segformer_b5.py",
                      epochs=ep + 1, seed=args.seed)
            print(f"  → guardado {args.out}")

    print(f"[OK] mejor mIoU SegFormer-B5 ({args.size}px): {best:.3f}")

    # Exporta EL MEJOR checkpoint (antes: última época) y con dynamo=False:
    # el default de torch 2.11 producía IR10 + 338 MB de pesos externos que ni
    # cv2.dnn ni el conversor IMX500 aceptan.
    from cansat.checkpoints import load_model_state
    base.load_state_dict(load_model_state(args.out))
    base.eval().cpu()
    wrap = Wrap(base, args.size).cpu()
    dst = f"outputs/cansat_seg_terrain_segformer_b5_{args.size}.onnx"
    torch.onnx.export(wrap, torch.randn(1, 3, args.size, args.size), dst,
                      input_names=["input"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True,
                      dynamo=False)
    print(f"[OK] {dst} (BEST, dynamo off)")


if __name__ == "__main__":
    main()
