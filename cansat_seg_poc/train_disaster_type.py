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
    # Tier 3 (2026-09-20, Kaggle xview2 tier-3-data): llena la clase tornado
    # (0 -> 3 eventos), suma volcán, Nepal/Sunda y 3 incendios.
    "joplin-tornado": "tornado", "moore-tornado": "tornado",
    "tuscaloosa-tornado": "tornado", "lower-puna-volcano": "volcan",
    "nepal-flooding": "inundacion", "sunda-tsunami": "inundacion",
    "pinery-bushfire": "incendio", "portugal-wildfire": "incendio",
    "woolsey-fire": "incendio",
}
# Subcadenas del nombre del ortomosaico CRASAR → tipo de desastre.
CRASAR_TIPO = {
    "MussettBayouFire": "incendio", "Kilauea": "volcan", "Geothermal": "volcan",
    "Mayfield": "tornado", "Steinhatchee": "huracan", "Cocodrie": "huracan",
    "Jena": "huracan", "0827": "huracan", "DMS": "huracan",
    "Champlain": "otro",
}


def recolectar_eventos(max_por_clase: int = 1200,
                       seed: int = 42) -> list[tuple[str, int, str]]:
    """(ruta_imagen, clase, evento) desde todos los manifests disponibles.

    El EVENTO (nombre del desastre/fuente) es lo que permite evaluar
    leave-one-event-out: sin él, un split aleatorio deja tiles del mismo evento
    en train y val y la accuracy sale inflada (fuga de evento).
    """
    rng = random.Random(seed)
    por_clase: dict[int, list[tuple[str, str]]] = {i: [] for i in range(len(CLASES))}

    xbd = ROOT / "dataset/xbd_masks/manifest.csv"
    if xbd.is_file():
        for r in csv.DictReader(xbd.open(encoding="utf-8")):
            ev = r["name"].split("_", 1)[0]
            tipo = XBD_EVENTO.get(ev)
            if tipo:
                por_clase[TIPO[tipo]].append((r["image"], f"xbd:{ev}"))

    kate = ROOT / "dataset/kate_pd_tiles/manifest_train.csv"
    if kate.is_file():
        for r in csv.DictReader(kate.open(encoding="utf-8")):
            por_clase[TIPO["sismo"]].append((r["image"], "kate_pd:turkiye"))

    rescue = ROOT / "dataset/rescuenet_tiles/manifest_train_sub8000.csv"
    if rescue.is_file():
        for r in csv.DictReader(rescue.open(encoding="utf-8")):
            por_clase[TIPO["huracan"]].append((r["image"], "rescuenet:huracan"))

    crasar = ROOT / "dataset/crasar_tiles/manifest_train.csv"
    crasar_test = ROOT / "dataset/crasar_tiles/manifest_test.csv"
    for man in (crasar, crasar_test):
        if not man.is_file():
            continue
        for r in csv.DictReader(man.open(encoding="utf-8")):
            hit = next((sub for sub in CRASAR_TIPO if sub in r["name"]), None)
            if hit:
                por_clase[TIPO[CRASAR_TIPO[hit]]].append((r["image"], f"crasar:{hit}"))

    filas: list[tuple[str, int, str]] = []
    for cls, pares in por_clase.items():
        rng.shuffle(pares)
        for p, ev in pares[:max_por_clase]:
            filas.append((p, cls, ev))
    rng.shuffle(filas)
    return filas


def recolectar(max_por_clase: int = 1200, seed: int = 42) -> list[tuple[str, int]]:
    """(ruta_imagen, clase) desde todos los manifests disponibles."""
    return [(p, c) for p, c, _e in recolectar_eventos(max_por_clase, seed)]


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


def _entrenar_fold(train, test, args, device, epochs=6):
    """Entrena un modelo en ``train`` y lo evalúa en ``test`` (un evento).

    Devuelve (acc, preds, reales). Sin checkpoints: es una evaluación LOEO,
    no un modelo de producción.
    """
    import collections

    model = modelo(len(CLASES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dl = DataLoader(TipoDS(train, args.size, aug=True), batch_size=args.batch,
                    shuffle=True, num_workers=4, persistent_workers=True)
    for _ in range(epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    model.eval()
    te_dl = DataLoader(TipoDS(test, args.size), batch_size=args.batch,
                       num_workers=4)
    correct = total = 0
    preds: collections.Counter = collections.Counter()
    reales: collections.Counter = collections.Counter()
    filas_probs: list[tuple[int, list[float]]] = []  # (y, probs de las 7 clases)
    with torch.no_grad():
        for x, y in te_dl:
            pr = torch.softmax(model(x.to(device)), dim=1).cpu()
            p = pr.argmax(1)
            correct += int((p == y).sum())
            total += len(y)
            for pi, yi, probs in zip(p.tolist(), y.tolist(), pr.tolist(), strict=True):
                preds[CLASES[pi]] += 1
                reales[CLASES[yi]] += 1
                filas_probs.append((yi, probs))
    return correct / max(1, total), preds, reales, filas_probs


def loeo(args) -> int:
    """Leave-one-event-out: la métrica honesta del clasificador de tipo.

    Con split aleatorio, tiles del mismo evento caen en train y val (fuga) y la
    accuracy sale inflada. Acá se entrena SIN el evento y se evalúa SOLO en él.
    """
    import json

    filas = recolectar_eventos(args.max_por_clase, args.seed)
    eventos = sorted({e for _p, _c, e in filas})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LOEO: {len(eventos)} eventos, {len(filas)} muestras")
    idx_fuego = CLASES.index("incendio")
    res: list[dict] = []
    filas_csv: list[tuple[str, int, float]] = []
    filas_todas: list[tuple[str, int, list[float]]] = []
    for ev in eventos:
        test = [(p, c) for p, c, e in filas if e == ev]
        train = [(p, c) for p, c, e in filas if e != ev]
        if len(test) < 30 or len(train) < 200:
            print(f"[skip] {ev}: test {len(test)} / train {len(train)}")
            continue
        acc, preds, reales, probs = _entrenar_fold(train, test, args, device)
        for yi, vector in probs:
            p_fuego = vector[idx_fuego]
            filas_csv.append((ev, 1 if CLASES[yi] == "incendio" else 0, p_fuego))
            filas_todas.append((ev, yi, vector))
        dom = max(reales, key=reales.get)
        pmay = max(preds, key=preds.get)
        res.append({"evento": ev, "n_test": len(test), "acc": round(acc, 4),
                    "clase_dominante": dom, "pred_mayoria": pmay})
        print(f"  {ev}: acc {acc:.3f} (n={len(test)}, dom {dom} → {pmay})")
    if res:
        w = sum(r["n_test"] for r in res)
        media = sum(r["acc"] * r["n_test"] for r in res) / w
        print(f"[OK] LOEO media ponderada: {media:.3f} "
              f"({len(res)} eventos, {w} tiles)")
        out = ROOT / "outputs/disaster_type_loeo.json"
        out.write_text(json.dumps(
            {"media_ponderada": round(media, 4), "eventos": res},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {out}")
        if args.guardar_probs and filas_csv:
            import csv as _csv
            with Path(args.guardar_probs).open("w", newline="",
                                               encoding="utf-8") as fh:
                wcsv = _csv.writer(fh)
                wcsv.writerow(["evento", "es_incendio", "p_incendio"])
                for ev, es, pf in filas_csv:
                    wcsv.writerow([ev, es, f"{pf:.6f}"])
            print(f"[OK] {args.guardar_probs} ({len(filas_csv)} tiles)")
        if args.probs_todas and filas_todas:
            import csv as _csv
            with Path(args.probs_todas).open("w", newline="",
                                             encoding="utf-8") as fh:
                wcsv = _csv.writer(fh)
                wcsv.writerow(["evento", "y"] + [f"p_{c}" for c in CLASES])
                for ev, yi, vector in filas_todas:
                    wcsv.writerow([ev, yi] + [f"{v:.6f}" for v in vector])
            print(f"[OK] {args.probs_todas} ({len(filas_todas)} tiles)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Clasificador de tipo de desastre")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--loeo", action="store_true",
                    help="evaluar leave-one-event-out en vez de entrenar el "
                         "modelo final (métrica honesta, ~6 épocas por fold)")
    ap.add_argument("--guardar-probs", default=None,
                    help="con --loeo: CSV (evento,es_incendio,p_incendio) por "
                         "tile para calibrar el umbral conformal "
                         "(cansat/conformal.py)")
    ap.add_argument("--probs-todas", default=None,
                    help="con --loeo: CSV (evento,y,p_<clase>...) con las 7 "
                         "probabilidades por tile, para ECE/temperatura y sets "
                         "adaptativos (cansat/calibracion.py)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-por-clase", type=int, default=1200)
    ap.add_argument("--out", default="outputs/best_disaster_type.pth")
    ap.add_argument("--onnx-out", default="outputs/cansat_disaster_type.onnx")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    if args.loeo:
        return loeo(args)

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
