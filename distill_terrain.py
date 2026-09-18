"""
CanSat La Base — Destilación SegFormer-B5 → DeepLabV3+ MobileNetV2.
Mejora el modelo de VUELO sin cambiar tamaño ni velocidad.

Uso:
    python distill_terrain.py --epochs 8 --batch 4
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation

from train import DeepLabV3PlusMobileNetV2
from train_terrain_v2 import conf_matrix

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLASSES = ["vege", "buil", "wate", "bare", "othe"]
# Nombres canónicos para el registro de modelos (cansat.indices).
IDX_CLASS_NAMES = ("vegetation", "building", "water", "bare_ground", "other")


class LoveDA(Dataset):
    def __init__(self, root, split, size=320, aug=False):
        root = Path(root)

        # Acepta "train"/"val" y lo convierte a la estructura real
        split_name = {
            "train": "Train",
            "val": "Val",
            "valid": "Val",
            "validation": "Val",
            "Train": "Train",
            "Val": "Val",
        }[split]

        self.pairs = []
        for domain in ["Rural", "Urban"]:
            img_dir = root / split_name / domain / "images_png"
            msk_dir = root / split_name / domain / "masks_png"

            imgs = sorted(img_dir.glob("*.png"))
            for ip in imgs:
                mp = msk_dir / ip.name
                if mp.exists():
                    self.pairs.append((ip, mp))

        self.size = size
        self.aug = aug

        print(f"[LoveDA] {split_name}: {len(self.pairs)} pares")

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No encontré pares en {root / split_name}. "
                "Revisá estructura images_png/masks_png."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]

        img = cv2.imread(str(ip))
        msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"No pude leer imagen: {ip}")
        if msk is None:
            raise RuntimeError(f"No pude leer mask: {mp}")

        img = cv2.resize(img, (self.size, self.size))
        msk = cv2.resize(msk, (self.size, self.size),
                         interpolation=cv2.INTER_NEAREST)

        if self.aug:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                msk = cv2.flip(msk, 1)
            if random.random() < 0.5:
                img = cv2.flip(img, 0)
                msk = cv2.flip(msk, 0)

            k = random.choice([0, 1, 2, 3])
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))
                msk = np.ascontiguousarray(np.rot90(msk, k))

            # Augmentación suave de brillo/contraste
            if random.random() < 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-12, 12)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).float()

        y = torch.from_numpy(msk.astype(np.int64))

        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=float, default=4.0, help="temperatura KD")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="peso hard loss (1=solo labels reales)")
    ap.add_argument("--seed", type=int, default=42)
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

    # Student = modelo de vuelo actual (acepta dict con metadata o crudo)
    from cansat.checkpoints import load_model_state
    student = DeepLabV3PlusMobileNetV2(5)
    student.load_state_dict(load_model_state("outputs/best_terrain_v2.pth"))
    student.to(device)

    # ⚠ Antes leía `class_weights.npy` de la RAÍZ: una caché vieja (16/8) con
    #   los mismos valores hardcodeados de train_qat.py. Si cambia el remapeo o
    #   el dataset, el destilado entrenaba con ponderación incorrecta y en
    #   silencio. Ahora se calculan del dataset actual, como train.py.
    from train import class_weights as compute_cw
    cw = compute_cw("dataset/loveda_remapped").to(device)
    print("Pesos de clase (calculados): " + ", ".join(f"{v:.2f}" for v in cw.tolist()))

    train_dl = DataLoader(LoveDA("dataset/loveda_remapped", "train", aug=True),
                          batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(LoveDA("dataset/loveda_remapped", "val"),
                        batch_size=args.batch, num_workers=4)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best = 0.0
    for ep in range(args.epochs):
        student.train()
        for x, y in train_dl:
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
                # ⚠ El KD se enmascara con ignore: antes el student imitaba al
                #   teacher TAMBIÉN en píxeles 255 (sin dato), donde el teacher
                #   no tiene supervisión. La máscara se aplica por píxel.
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
            for x, y in val_dl:
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
            from cansat.checkpoints import save_ckpt
            save_ckpt("outputs/best_terrain_v2_distilled.pth", student,
                      num_classes=5, img_size=320,
                      # CLASSES de este archivo son abreviados ("vege"...):
                      # el registro usa los nombres canónicos de cansat.indices.
                      class_names=list(IDX_CLASS_NAMES),
                      miou=round(float(miou), 4),
                      iou_per_class=[round(float(x), 4) for x in ious],
                      dataset="loveda_remapped", teacher="segformer_b5",
                      script="distill_terrain.py", epochs=ep + 1, seed=args.seed)
            tag = " → guardado"
        print(f"  epoch {ep+1}: mIoU {miou:.3f} | "
              + "  ".join(f"{CLASSES[i]}={ious[i]:.2f}" for i in range(5))
              + tag)

    print(f"[OK] mejor mIoU destilado: {best:.3f}")

    # Exporta EL MEJOR checkpoint y con dynamo=False (antes: última época y
    # dynamo default → IR10 + pesos externos, incompatibles con cv2.dnn/IMX500).
    student.load_state_dict(load_model_state("outputs/best_terrain_v2_distilled.pth"))
    student.eval().cpu()
    torch.onnx.export(student, torch.randn(1, 3, 320, 320),
                      "outputs/cansat_seg_terrain_v2_distilled.onnx",
                      input_names=["input"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True,
                      dynamo=False)
    print("[OK] outputs/cansat_seg_terrain_v2_distilled.onnx (BEST, dynamo off)")


if __name__ == "__main__":
    main()
