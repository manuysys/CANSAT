"""
UDA LoveDA -> RescueNet para el terreno (V11 4.3): baseline de self-training.

PoC honesto y time-boxed:
  1. El teacher (terreno v2 @224, ONNX) pseudo-etiqueta imágenes UAV de
     RescueNet train (sin GT de terreno) con filtro de confianza.
  2. El student (tiny de vuelo, LR-ASPP MV3-S @224) se fine-tunea sobre esas
     pseudo-etiquetas + un replay de LoveDA (para no olvidar el dominio fuente).
  3. Evaluación ANTES/DESPUÉS:
     · target (UAV): mIoU contra un PROXY de terreno construido desde los
       labels ORIGINALES de RescueNet val (1=agua, 2-5=edificio, 9=árbol,
       6/7/8/10=otro, 0=ignorar). Es un proxy débil, declarado.
     · fuente (LoveDA val, subconjunto): retención.

Uso:
    python tools/uda_terrain.py --n-train 300 --epochs 10

Salida: docs/benchmarks/uda_terrain.json + outputs/best_terrain_tiny_uda.pth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2                                                        # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
import torch.nn.functional as F                                   # noqa: E402
from torch.utils.data import DataLoader, Dataset                  # noqa: E402

from cansat.checkpoints import load_model_state, save_ckpt        # noqa: E402
from distill_terrain import MEAN, STD                             # noqa: E402
from train_terrain_tiny import build as build_tiny                # noqa: E402
from train_terrain_v2 import conf_matrix                          # noqa: E402

# Proxy de terreno desde las clases ORIGINALES de RescueNet (0..10):
#   0=fondo (ignorar), 1=agua, 2-5=edificio, 6=vehículo, 7/8=ruta,
#   9=árbol, 10=pileta. Terreno: 0=vege, 1=buil, 2=wate, 3=bare, 4=othe.
PROXY = {1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 4, 7: 4, 8: 4, 9: 0, 10: 4}
IGNORE = 255


def proxy_mask(label: np.ndarray) -> np.ndarray:
    out = np.full(label.shape, IGNORE, np.uint8)
    for src, dst in PROXY.items():
        out[label == src] = dst
    return out


def leer(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"no pude leer {path}")
    img = cv2.resize(img, (size, size))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - MEAN) / STD


class UAVTrain(Dataset):
    """Imágenes UAV (train) con pseudo-etiquetas del teacher (o None)."""

    def __init__(self, imgs: list[Path], size: int,
                 pseudo: list[np.ndarray] | None = None):
        self.imgs, self.size, self.pseudo = imgs, size, pseudo

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = leer(self.imgs[i], self.size).transpose(2, 0, 1)
        y = (self.pseudo[i] if self.pseudo is not None
             else np.full((self.size, self.size), IGNORE, np.uint8))
        return torch.from_numpy(x).float(), torch.from_numpy(y.astype(np.int64))


class UAVEval(Dataset):
    """Imágenes UAV (val) con el proxy de terreno como GT débil."""

    def __init__(self, imgs: list[Path], labels: list[Path], size: int):
        self.imgs, self.labels, self.size = imgs, labels, size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = leer(self.imgs[i], self.size).transpose(2, 0, 1)
        lab = cv2.imread(str(self.labels[i]), cv2.IMREAD_GRAYSCALE)
        lab = cv2.resize(lab, (self.size, self.size),
                         interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(x).float(), torch.from_numpy(
            proxy_mask(lab).astype(np.int64))


def _miou_proxy(cm: np.ndarray) -> tuple[float, dict]:
    """mIoU sobre las clases del proxy PRESENTES en la GT (0,1,2,4)."""
    ious = {}
    for i in range(5):
        inter = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - inter
        if cm[:, i].sum() > 0:
            ious[i] = round(float(inter / max(union, 1)), 4)
    return float(np.mean(list(ious.values()))) if ious else 0.0, ious


def evaluar(model, dl, dev) -> tuple[float, dict]:
    model.eval()
    cm = np.zeros((5, 5), np.int64)
    with torch.no_grad():
        for x, y in dl:
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                pred = model(x.to(dev)).argmax(1).cpu().numpy()
            cm += conf_matrix(pred, y.numpy())
    return _miou_proxy(cm)


def main() -> int:
    ap = argparse.ArgumentParser(description="UDA terreno (PoC)")
    ap.add_argument("--teacher-onnx", default="outputs/cansat_seg_terrain_v2_224.onnx")
    ap.add_argument("--student", default="outputs/best_terrain_tiny.pth")
    ap.add_argument("--rescuenet", default="dataset/rescuenet")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--conf", type=float, default=0.7,
                    help="confianza mínima del teacher para pseudo-etiquetar")
    ap.add_argument("--replay", type=float, default=0.25,
                    help="fracción de lotes de replay LoveDA por época")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true",
                    help="forzar CPU (p. ej. si la GPU está ocupada entrenando)")
    ap.add_argument("--out", default="docs/benchmarks/uda_terrain.json")
    args = ap.parse_args()

    from cansat.seed import set_seed
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                       else "cuda")
    print(f"Device: {dev}")

    root = Path(args.rescuenet)
    tr_imgs = sorted((root / "train-org-img").glob("*.jpg")) or \
        sorted((root / "train-org-img").glob("*.png"))
    va_imgs = sorted((root / "val-org-img").glob("*.jpg")) or \
        sorted((root / "val-org-img").glob("*.png"))
    va_labs = sorted((root / "val-label-img").glob("*.png"))
    if not tr_imgs or not va_imgs or not va_labs:
        print("[ERROR] estructura de RescueNet inesperada "
              "(train-org-img / val-org-img / val-label-img)")
        return 1
    tr_imgs = [tr_imgs[i] for i in rng.permutation(len(tr_imgs))[:args.n_train]]
    nv = min(args.n_val, len(va_imgs))
    va_idx = rng.permutation(len(va_imgs))[:nv]
    va_imgs = [va_imgs[i] for i in va_idx]
    va_labs = [va_labs[i] for i in va_idx]
    print(f"RescueNet: {len(tr_imgs)} train (pseudo) · {len(va_imgs)} val (proxy)")

    # 1) Teacher: pseudo-etiquetas con filtro de confianza.
    t0 = time.time()
    from cansat import onnxio
    teacher = onnxio.OnnxModel(args.teacher_onnx, required=True,
                               label="teacher-terreno-v2")
    pseudo: list[np.ndarray] = []
    for p in tr_imgs:
        x = leer(p, args.size).transpose(2, 0, 1)[None]
        logits = teacher.run({"input": x.astype(np.float32)})
        prob = torch.softmax(torch.from_numpy(logits), 1)[0].numpy()
        conf = prob.max(0)
        lab = prob.argmax(0).astype(np.uint8)
        lab[conf < args.conf] = IGNORE
        pseudo.append(lab)
    print(f"Pseudo-etiquetas: {len(pseudo)} imgs en {time.time() - t0:.0f} s "
          f"(conf >= {args.conf})")

    # 2) Student tiny: antes (base) y fine-tune con UDA + replay.
    student = build_tiny("lraspp-mv3s", 5, pretrained=False)
    student.load_state_dict(load_model_state(args.student))
    student.to(dev)

    val_dl = DataLoader(UAVEval(va_imgs, va_labs, args.size),
                        batch_size=args.batch, num_workers=0)
    antes_miou, antes_cls = evaluar(student, val_dl, dev)
    print(f"ANTES  (target proxy): mIoU {antes_miou:.4f} · {antes_cls}")

    from distill_terrain import LoveDA
    love = LoveDA("dataset/loveda_remapped", "val", size=args.size)
    love_dl = DataLoader(love, batch_size=args.batch, num_workers=0,
                         shuffle=True)
    love_it = iter(love_dl)
    train_dl = DataLoader(UAVTrain(tr_imgs, args.size, pseudo),
                          batch_size=args.batch, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    for ep in range(args.epochs):
        student.train()
        for x, y in train_dl:
            # Replay: un lote de LoveDA cada ~4 lotes UAV.
            if args.replay > 0 and rng.random() < args.replay:
                try:
                    x, y = next(love_it)
                except StopIteration:
                    love_it = iter(love_dl)
                    x, y = next(love_it)
            x, y = x.to(dev), y.to(dev)
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                out = student(x)
                loss = F.cross_entropy(out, y, ignore_index=IGNORE)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  época {ep + 1}/{args.epochs} · loss {loss.item():.3f}",
              flush=True)

    despues_miou, despues_cls = evaluar(student, val_dl, dev)
    print(f"DESPUÉS (target proxy): mIoU {despues_miou:.4f} · {despues_cls}")

    # Retención en la fuente: subconjunto de LoveDA val (rápido).
    from torch.utils.data import Subset
    sub = Subset(love, list(range(min(300, len(love)))))
    love_val = DataLoader(sub, batch_size=args.batch, num_workers=0)
    cm = np.zeros((5, 5), np.int64)
    student.eval()
    with torch.no_grad():
        for x, y in love_val:
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                pred = student(x.to(dev)).argmax(1).cpu().numpy()
            cm += conf_matrix(pred, y.numpy())
    ious = []
    for i in range(5):
        inter = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - inter
        ious.append(inter / max(union, 1))
    fuente_miou = float(np.mean(ious))
    print(f"Fuente (LoveDA val 300): mIoU {fuente_miou:.4f}")

    res = {
        "protocolo": {
            "teacher": args.teacher_onnx,
            "student": args.student,
            "n_train_ucl": len(tr_imgs), "n_val_proxy": len(va_imgs),
            "conf_pseudo": args.conf, "epochs": args.epochs,
            "replay": args.replay, "lr": args.lr,
            "proxy": "RescueNet original: 1=agua, 2-5=edificio, 9=arbol, 6/7/8/10=otro, 0=ignorar (proxy DEBIL declarado)",
        },
        "target_proxy_antes": {"miou": round(antes_miou, 4), "por_clase": antes_cls},
        "target_proxy_despues": {"miou": round(despues_miou, 4), "por_clase": despues_cls},
        "fuente_loveda_val300": round(fuente_miou, 4),
        "delta_target": round(despues_miou - antes_miou, 4),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    save_ckpt("outputs/best_terrain_tiny_uda.pth", student,
              num_classes=5, img_size=args.size,
              class_names=("vegetation", "building", "water", "bare_ground",
                           "other"),
              miou=round(despues_miou, 4), dataset="loveda+rescuenet_pseudo",
              script="tools/uda_terrain.py", epochs=args.epochs,
              seed=args.seed)
    student.eval().cpu()
    torch.onnx.export(student, torch.randn(1, 3, args.size, args.size),
                      "outputs/cansat_seg_terrain_tiny_224_uda.onnx",
                      input_names=["input"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print("[OK] outputs/best_terrain_tiny_uda.pth + "
          "outputs/cansat_seg_terrain_tiny_224_uda.onnx")
    print(f"  Δ target proxy: {res['delta_target']:+.4f} · "
          f"fuente {fuente_miou:.4f}")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
