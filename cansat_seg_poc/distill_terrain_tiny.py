"""
CanSat La Base — Destilación SegFormer-B5 → tiny de VUELO (LR-ASPP MV3-S @224).

Mejora el modelo que VUELA sin cambiar arquitectura, tamaño ni velocidad: el
student es el mismo `LRASPPMobileNetV3Small` de train_terrain_tiny.py y se
exporta a 224 con dynamo=False (compatible cv2.dnn / IMX500).

Uso:
    python distill_terrain_tiny.py --epochs 8 --batch 8
    python distill_terrain_tiny.py --limit-batches 3   # smoke test
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from cansat.checkpoints import load_model_state, save_ckpt
from distill_terrain import CLASSES, IDX_CLASS_NAMES, LoveDA
from train import class_weights as compute_cw
from train_terrain_tiny import build as build_tiny
from train_terrain_v2 import conf_matrix

SIZE = 224


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=float, default=4.0, help="temperatura KD")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="peso hard loss (1=solo labels reales)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-batches", type=int, default=0,
                    help="corta train/val a N lotes (smoke test)")
    args = ap.parse_args()

    from cansat.seed import set_seed
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Teacher B5 congelado
    teacher = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640", num_labels=5,
        ignore_mismatched_sizes=True)
    teacher.load_state_dict(torch.load("outputs/best_terrain_segformer_b5.pth",
                                       map_location="cpu"))
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student = tiny de vuelo actual
    student = build_tiny("lraspp-mv3s", 5, pretrained=False)
    student.load_state_dict(load_model_state("outputs/best_terrain_tiny.pth"))
    student.to(device)

    cw = compute_cw("dataset/loveda_remapped").to(device)
    print("Pesos de clase (calculados): " + ", ".join(f"{v:.2f}" for v in cw.tolist()))

    train_dl = DataLoader(LoveDA("dataset/loveda_remapped", "train", size=SIZE,
                                 aug=True),
                          batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(LoveDA("dataset/loveda_remapped", "val", size=SIZE),
                        batch_size=args.batch, num_workers=4)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best = 0.0
    for ep in range(args.epochs):
        student.train()
        for i, (x, y) in enumerate(train_dl):
            if args.limit_batches and i >= args.limit_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                lt = F.interpolate(teacher(x).logits, size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)
            with torch.amp.autocast("cuda"):
                ls = student(x)
                if ls.shape[-2:] != x.shape[-2:]:
                    ls = F.interpolate(ls, size=x.shape[-2:], mode="bilinear",
                                       align_corners=False)
                hard = F.cross_entropy(ls, y, weight=cw, ignore_index=255)
                # KD enmascarado con ignore (igual que distill_terrain.py).
                kd_map = F.kl_div(F.log_softmax(ls / args.T, 1),
                                  F.softmax(lt / args.T, 1),
                                  reduction="none").sum(1) * (args.T ** 2)
                valido = (y != 255)
                kd = kd_map[valido].mean() if valido.any() else kd_map.mean()
                loss = args.alpha * hard + (1 - args.alpha) * kd
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        student.eval()
        cm = np.zeros((5, 5), dtype=np.int64)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_dl):
                if args.limit_batches and i >= args.limit_batches:
                    break
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda"):
                    ls = student(x)
                cm += conf_matrix(ls.argmax(1).cpu().numpy(), y.cpu().numpy())
        ious = []
        for i in range(5):
            inter = cm[i, i]
            union = cm[i, :].sum() + cm[:, i].sum() - inter
            ious.append(inter / max(union, 1))
        miou = float(np.mean(ious))
        tag = ""
        if miou > best:
            best = miou
            save_ckpt("outputs/best_terrain_tiny_distilled.pth", student,
                      num_classes=5, img_size=SIZE,
                      class_names=list(IDX_CLASS_NAMES),
                      miou=round(float(miou), 4),
                      iou_per_class=[round(float(v), 4) for v in ious],
                      dataset="loveda_remapped", teacher="segformer_b5",
                      script="distill_terrain_tiny.py", epochs=ep + 1,
                      seed=args.seed)
            tag = " → guardado"
        print(f"  epoch {ep+1}: mIoU {miou:.3f} | "
              + "  ".join(f"{CLASSES[i]}={ious[i]:.2f}" for i in range(5))
              + tag, flush=True)

    print(f"[OK] mejor mIoU destilado (tiny): {best:.3f}")

    if not args.limit_batches:
        # Exporta EL MEJOR checkpoint a 224 (dynamo=False: compatible cv2.dnn).
        student.load_state_dict(
            load_model_state("outputs/best_terrain_tiny_distilled.pth"))
        student.eval().cpu()
        torch.onnx.export(student, torch.randn(1, 3, SIZE, SIZE),
                          "outputs/cansat_seg_terrain_tiny_224_distilled.onnx",
                          input_names=["input"], output_names=["logits"],
                          opset_version=17, do_constant_folding=True,
                          dynamo=False)
        print("[OK] outputs/cansat_seg_terrain_tiny_224_distilled.onnx "
              "(BEST, dynamo off)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
