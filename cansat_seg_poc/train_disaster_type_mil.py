"""
train_disaster_type_mil.py - Clasificador de tipo con MIL (Multiple Instance
Learning) por evento: la BOLSA es el evento, las INSTANCIAS son sus tiles.

Motivación honesta: el clasificador por tile no generaliza a eventos nuevos
(LOEO 0.3681 ponderado; el 0.978 era fuga de evento). La etiqueta de tipo es
del EVENTO, no del tile: MIL es la formulación correcta. Acá se entrena un
backbone por tile + pooling con ATENCIÓN sobre los tiles de la bolsa, y se
evalúa LOEO (leave-one-event-out) a nivel BOLSA, comparable con el baseline
por tile de outputs/disaster_type_loeo.json.

Uso:
    python train_disaster_type_mil.py --epochs 6 --k 8 --max-bolsas 40
    python train_disaster_type_mil.py --loeo          # evalúa todos los folds
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_disaster_type import CLASES, MEAN, STD, recolectar_eventos

ROOT = Path(__file__).resolve().parent


class BolsasDS(Dataset):
    """Cada ítem es una bolsa: K tiles de un evento + la clase del evento."""

    def __init__(self, por_evento: dict[str, list[tuple[str, int]]],
                 k: int = 8, size: int = 224, max_bolsas: int = 40,
                 seed: int = 42, aug: bool = False):
        rng = random.Random(seed)
        self.bolsas: list[tuple[list[str], int]] = []
        for ev, tiles in sorted(por_evento.items()):
            if not tiles:
                continue
            cls = tiles[0][1]
            rutas = [p for p, _c in tiles]
            n = max(1, min(max_bolsas, len(rutas) // max(1, k // 2)))
            for _ in range(n):
                sel = rng.sample(rutas, min(k, len(rutas)))
                self.bolsas.append((sel, cls))
        rng.shuffle(self.bolsas)
        self.k, self.size, self.aug = k, size, aug

    def __len__(self) -> int:
        return len(self.bolsas)

    def _tile(self, p: str) -> torch.Tensor:
        img = cv2.imread(p)
        img = cv2.resize(img, (self.size, self.size))
        if self.aug and random.random() < 0.5:
            img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        return torch.from_numpy(rgb.transpose(2, 0, 1)).float()

    def __getitem__(self, i):
        rutas, cls = self.bolsas[i]
        return torch.stack([self._tile(p) for p in rutas]), cls


class MILTipo(torch.nn.Module):
    """MobileNetV3-Small por tile + atención gatada sobre la bolsa."""

    def __init__(self, n_clases: int):
        super().__init__()
        from torchvision.models import (MobileNet_V3_Small_Weights,
                                        mobilenet_v3_small)
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = m.features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        dim = 576
        self.at_a = torch.nn.Linear(dim, 128)
        self.at_b = torch.nn.Linear(dim, 128)
        self.head = torch.nn.Linear(dim, n_clases)

    def forward(self, x):  # x: (B, K, 3, S, S)
        b, k = x.shape[:2]
        h = self.pool(self.features(x.view(b * k, *x.shape[2:]))).flatten(1)
        h = h.view(b, k, -1)
        w = torch.exp(self.at_a(h)) * torch.tanh(self.at_b(h))
        w = torch.softmax(w.sum(-1), dim=1)          # (B, K)
        z = (h * w.unsqueeze(-1)).sum(1)             # (B, dim)
        return self.head(z), w


def folds(filas, k: int, size: int, max_bolsas: int, seed: int):
    por_evento: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for p, c, ev in filas:
        por_evento[ev].append((p, c))
    return por_evento


def entrenar_fold(por_evento, ev_test: str, args, device):
    train_ev = {e: t for e, t in por_evento.items() if e != ev_test}
    test_ev = {ev_test: por_evento[ev_test]}
    ds_tr = BolsasDS(train_ev, args.k, args.size, args.max_bolsas,
                     args.seed, aug=True)
    ds_te = BolsasDS(test_ev, args.k, args.size, 20, args.seed, aug=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers)
    dl_te = DataLoader(ds_te, batch_size=args.batch, num_workers=args.workers)
    model = MILTipo(len(CLASES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for _ in range(args.epochs):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x)[0], y).backward()
            opt.step()
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for x, y in dl_te:
            pred = model(x.to(device))[0].argmax(1).cpu()
            ok += int((pred == y).sum())
            tot += len(y)
    return ok / max(1, tot), tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--k", type=int, default=8, help="tiles por bolsa")
    ap.add_argument("--max-bolsas", type=int, default=40,
                    help="tope de bolsas por evento")
    ap.add_argument("--batch", type=int, default=8, help="bolsas por paso")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-por-clase", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loeo", action="store_true",
                    help="evaluar leave-one-event-out (todos los folds)")
    ap.add_argument("--out", default="outputs/mil_loeo.json")
    args = ap.parse_args()

    filas = recolectar_eventos(args.max_por_clase, args.seed)
    por_evento = folds(filas, args.k, args.size, args.max_bolsas, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eventos = sorted(por_evento)
    print(f"MIL: {len(eventos)} eventos, {len(filas)} tiles, "
          f"K={args.k}, tope {args.max_bolsas} bolsas/evento")

    if not args.loeo:
        print("Entrená el modelo final con --loeo desactivado no está "
              "implementado: la evidencia honesta es el LOEO.")
        return 1

    res = []
    for ev in eventos:
        if len(por_evento[ev]) < 30:
            print(f"[skip] {ev}: {len(por_evento[ev])} tiles")
            continue
        acc, n = entrenar_fold(por_evento, ev, args, device)
        res.append({"evento": ev, "n_test_tiles": len(por_evento[ev]),
                    "bolsas_test": n, "acc_bolsa": round(acc, 4)})
        print(f"  {ev}: acc bolsa {acc:.3f} ({n} bolsas)")
    if res:
        w = sum(r["bolsas_test"] for r in res)
        media = sum(r["acc_bolsa"] * r["bolsas_test"] for r in res) / w
        base = None
        pj = ROOT / "outputs/disaster_type_loeo.json"
        if pj.is_file():
            d = json.loads(pj.read_text(encoding="utf-8"))
            base = d.get("media_ponderada")
        out = {
            "media_ponderada_bolsa": round(media, 4),
            "baseline_por_tile_loeo": base,
            "eventos": res,
            "nota": ("MIL a nivel bolsa (evento) con atención sobre tiles; "
                     "comparable con el baseline por tile del LOEO. K="
                     f"{args.k}, {args.epochs} épocas, tope "
                     f"{args.max_bolsas} bolsas/evento."),
        }
        Path(args.out).write_text(json.dumps(out, indent=2,
                                             ensure_ascii=False),
                                  encoding="utf-8")
        print(f"[OK] MIL media bolsa {media:.3f} vs tile {base} · {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
