"""
CanSat La Base — F2 v3b: clasificador de TIPO de desastre (7 clases).

Responde "¿qué desastre es?" a partir del frame, con las etiquetas de EVENTO
como supervisión débil (el evento no siempre se ve en un tile: es una pista,
no una verdad). Clases:

    0 huracan · 1 inundacion · 2 sismo · 3 incendio · 4 volcan · 5 tornado · 6 otro

Fuentes (todas ya descargadas): xBD (10 eventos), RescueNet (huracán),
KATE-PD (sismo Türkiye) y CRASAR (10 desastres sUAS: tornado, volcán,
incendio, huracanes, colapso).

Modelo: MobileNetV3-Small (~2.5 M params, ONNX <10 MB, candidato NPU).

Uso:
    python train_disaster_type.py --epochs 12
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2                                                          # noqa: E402
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import torch.nn.functional as F                                     # noqa: E402
from torch.utils.data import DataLoader, Dataset                    # noqa: E402

from cansat.checkpoints import save_ckpt                            # noqa: E402
from cansat.seed import set_seed                                    # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASES = ["huracan", "inundacion", "sismo", "incendio", "volcan", "tornado",
          "otro"]
TIPO = {c: i for i, c in enumerate(CLASES)}

XBD_EVENTO = {
    "hurricane-michael": "huracan", "hurricane-florence": "huracan",
    "hurricane-harvey": "huracan", "hurricane-matthew": "huracan",
    "midwest-flooding": "inundacion", "palu-tsunami": "inundacion",
    "mexico-earthquake": "sismo", "guatemala-volcano": "volcan",
    "socal-fire": "incendio", "santa-rosa-wildfire": "incendio",
}
# Subcadenas del nombre del ortomosaico CRASAR → tipo de desastre.
CRASAR_TIPO = {
    "MussettBayouFire": "incendio", "Kilauea": "volcan", "Geothermal": "volcan",
    "Mayfield": "tornado", "Steinhatchee": "huracan", "Cocodrie": "huracan",
    "Jena": "huracan", "0827": "huracan", "DMS": "huracan",
    "Champlain": "otro",
}


def recolectar(max_por_clase: int = 1200, seed: int = 42) -> list[tuple[str, int]]:
    """(ruta_imagen, clase) desde todos los manifests disponibles."""
    rng = random.Random(seed)
    por_clase: dict[int, list[str]] = {i: [] for i in range(len(CLASES))}

    xbd = ROOT / "dataset/xbd_masks/manifest.csv"
    if xbd.is_file():
        for r in csv.DictReader(xbd.open(encoding="utf-8")):
            ev = r["name"].split("_", 1)[0]
            tipo = XBD_EVENTO.get(ev)
            if tipo:
                por_clase[TIPO[tipo]].append(r["image"])

    kate = ROOT / "dataset/kate_pd_tiles/manifest_train.csv"
    if kate.is_file():
        for r in csv.DictReader(kate.open(encoding="utf-8")):
            por_clase[TIPO["sismo"]].append(r["image"])

    rescue = ROOT / "dataset/rescuenet_tiles/manifest_train_sub8000.csv"
    if rescue.is_file():
        for r in csv.DictReader(rescue.open(encoding="utf-8")):
            por_clase[TIPO["huracan"]].append(r["image"])

    crasar = ROOT / "dataset/crasar_tiles/manifest_train.csv"
    if crasar.is_file():
        for r in csv.DictReader(crasar.open(encoding="utf-8")):
            tipo = next((t for sub, t in CRASAR_TIPO.items()
                         if sub in r["name"]), None)
            if tipo:
                por_clase[TIPO[tipo]].append(r["image"])

    filas: list[tuple[str, int]] = []
    for cls, rutas in por_clase.items():
        rng.shuffle(rutas)
        for p in rutas[:max_por_clase]:
            filas.append((p, cls))
    rng.shuffle(filas)
    return filas


class TipoDS(Dataset):
    def __init__(self, filas, size=224, aug=False):
        self.filas, self.size, self.aug = filas, size, aug

    def __len__(self):
        return len(self.filas)

    def __getitem__(self, i):
        p, cls = self.filas[i]
        img = cv2.imread(p)
        img = cv2.resize(img, (self.size, self.size))
        if self.aug and random.random() < 0.5:
            img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        return (torch.from_numpy(rgb.transpose(2, 0, 1)).float(),
                torch.tensor(cls, dtype=torch.long))


def modelo(n_clases: int):
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, n_clases)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="Clasificador de tipo de desastre")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-por-clase", type=int, default=1200)
    ap.add_argument("--out", default="outputs/best_disaster_type.pth")
    ap.add_argument("--onnx-out", default="outputs/cansat_disaster_type.onnx")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    filas = recolectar(args.max_por_clase, args.seed)
    import collections
    cuenta = collections.Counter(c for _p, c in filas)
    print("Muestras por clase: " +
          ", ".join(f"{CLASES[c]}={n}" for c, n in sorted(cuenta.items())))
    # Split estratificado 80/20.
    rng = random.Random(args.seed)
    idx = list(range(len(filas)))
    rng.shuffle(idx)
    n_val = max(1, int(len(filas) * 0.2))
    val = [filas[i] for i in idx[:n_val]]
    train = [filas[i] for i in idx[n_val:]]
    print(f"train {len(train)} / val {len(val)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = modelo(len(CLASES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_dl = DataLoader(TipoDS(train, args.size, aug=True),
                          batch_size=args.batch, shuffle=True, num_workers=4,
                          persistent_workers=True)
    val_dl = DataLoader(TipoDS(val, args.size), batch_size=args.batch,
                        num_workers=4, persistent_workers=True)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += int((pred == y).sum())
                total += len(y)
        acc = correct / max(1, total)
        print(f"  epoch {ep + 1}: acc val {acc:.3f} | loss {run / len(train_dl):.4f}")
        if acc > best:
            best = acc
            save_ckpt(args.out, model, num_classes=len(CLASES),
                      img_size=args.size, class_names=CLASES,
                      acc_val=round(acc, 4),
                      dataset="xBD + RescueNet + KATE-PD + CRASAR (etiqueta de evento)",
                      script="train_disaster_type.py",
                      epochs=ep + 1, seed=args.seed)
            print(f"  → guardado {args.out}")
    print(f"[OK] mejor acc: {best:.3f}")

    from cansat.checkpoints import load_model_state
    model.load_state_dict(load_model_state(args.out, strict=True))
    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, args.size, args.size),
                      args.onnx_out, input_names=["input"],
                      output_names=["logits"], opset_version=17,
                      do_constant_folding=True, dynamo=False)
    print(f"[OK] {args.onnx_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
