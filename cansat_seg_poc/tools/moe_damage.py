"""
MoE de daño (V11 4.2): compuerta APRENDIDA sobre los especialistas de daño.

PoC honesto: dos expertos (principal xBD y two-stage UAV) + una compuerta que
elige por tile cuál usar según features baratas (probabilidades de cada experto,
desacuerdo entre ellos, brillo/contraste de la imagen). Se entrena con GT en un
split y se evalúa en otro; se compara contra:

  · A siempre / B siempre (los especialistas solos),
  · promedio de probabilidades (ensemble clásico),
  · compuerta aprendida (logística sobre features),
  · oráculo (techo: el mejor por tile con GT).

Uso:
    python tools/moe_damage.py --cpu \
        --experto-a outputs/best_damage3.pth \
        --experto-b outputs/best_damage_v3_bal.pth \
        --manifest-a dataset/xbd_masks/manifest.csv \
        --manifest-b dataset/rescuenet_tiles/manifest_val.csv

Salida: docs/benchmarks/moe_damage.json + resumen por consola.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
from torch.utils.data import DataLoader                             # noqa: E402

from cansat.checkpoints import load_into                            # noqa: E402
from cansat.moe import FEATURES                                     # noqa: E402
from cansat.xbd import split_por_desastre                            # noqa: E402
from train import DeepLabV3PlusMobileNetV2                          # noqa: E402
from train_damage_v3 import XBDv3                                    # noqa: E402


def _iou_two_stage(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU de dañado two-stage (sólo sobre edificios: GT >= 1)."""
    pd, gd, gb = pred == 2, gt == 2, gt >= 1
    inter = int((pd & gd).sum())
    union = int(((pd & gb) | gd).sum())
    return inter / union if union else 0.0


def _cargar_modelo(ckpt: str, dev: torch.device) -> torch.nn.Module:
    model = DeepLabV3PlusMobileNetV2(3)
    load_into(model, ckpt, strict=False, min_loaded_frac=0.5)
    return model.to(dev).eval()


def recolectar(rows: list[dict], size: int, batch: int, dev: torch.device,
               m_a: torch.nn.Module, m_b: torch.nn.Module,
               max_tiles: int = 0) -> list[dict]:
    """Corre ambos expertos sobre las filas y devuelve features + IoUs."""
    if max_tiles:
        rows = rows[:max_tiles]
    dl = DataLoader(XBDv3(rows, size), batch_size=batch, num_workers=0)
    out: list[dict] = []
    with torch.no_grad():
        for x, y in dl:
            xg = x.to(dev)
            pa = torch.softmax(m_a(xg), 1).cpu().numpy()
            pb = torch.softmax(m_b(xg), 1).cpu().numpy()
            yt = y.numpy()
            for i in range(pa.shape[0]):
                p_a, p_b = pa[i], pb[i]
                gt = yt[i]
                ia = _iou_two_stage(p_a.argmax(0), gt)
                ib = _iou_two_stage(p_b.argmax(0), gt)
                img = x[i].numpy().transpose(1, 2, 0)
                brillo = float(img.mean())
                contraste = float(img.std())
                out.append({
                    "iou_a": ia, "iou_b": ib,
                    "mejor": int(ib > ia),          # etiqueta de la compuerta
                    "feats": [
                        float(p_a[2].mean()), float(p_a[2].max()),
                        float(p_b[2].mean()), float(p_b[2].max()),
                        float(np.abs(p_a[2] - p_b[2]).mean()),
                        brillo, contraste,
                    ],
                    "pa": p_a, "pb": p_b, "gt": gt,
                })
    return out


def _iou_estrategia(items: list[dict], pick) -> float:
    """IoU medio de la estrategia ``pick(item) -> prob map``."""
    vals = []
    for it in items:
        pred = pick(it).argmax(0)
        vals.append(_iou_two_stage(pred, it["gt"]))
    return float(np.mean(vals)) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="MoE de daño (PoC)")
    ap.add_argument("--experto-a", default="outputs/best_damage3.pth")
    ap.add_argument("--experto-b", default="outputs/best_damage_v3_bal.pth")
    ap.add_argument("--manifest-a", default="dataset/xbd_masks/manifest.csv")
    ap.add_argument("--manifest-b",
                    default="dataset/rescuenet_tiles/manifest_val.csv")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-por-manifest", type=int, default=150)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="docs/benchmarks/moe_damage.json")
    ap.add_argument("--guardar-gate", default="outputs/moe_gate.json",
                    help="artefacto JSON del gate para el post-vuelo "
                         "(lo consume cansat/moe.py)")
    args = ap.parse_args()

    from cansat.seed import set_seed
    set_seed(args.seed)

    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                       else "cuda")
    print(f"Device: {dev}")

    # Rows: A = holdout por desastre del xBD (eventos que no entrenaron),
    # B = val de RescueNet (UAV). El mismo dataset/loss que la evaluación.
    rows_a_all = list(csv.DictReader(open(args.manifest_a, encoding="utf-8")))
    _tr, rows_a, grupos = split_por_desastre(rows_a_all, args.holdout_frac,
                                             args.seed)
    rows_b = list(csv.DictReader(open(args.manifest_b, encoding="utf-8")))
    print(f"xBD held-out: {len(rows_a)} ({grupos}) · RescueNet val: {len(rows_b)}")

    m_a = _cargar_modelo(args.experto_a, dev)
    m_b = _cargar_modelo(args.experto_b, dev)

    t0 = time.time()
    items = recolectar(rows_a, args.size, args.batch, dev, m_a, m_b,
                       args.max_por_manifest)
    items += recolectar(rows_b, args.size, args.batch, dev, m_a, m_b,
                        args.max_por_manifest)
    print(f"{len(items)} tiles evaluados en {time.time() - t0:.0f} s")

    # Split train/test por tile (semilla fija).
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(items))
    n_tr = int(0.5 * len(idx))
    train = [items[i] for i in idx[:n_tr]]
    test = [items[i] for i in idx[n_tr:]]

    # Compuerta: regresión logística sobre las features (sklearn).
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr = np.array([it["feats"] for it in train])
    ytr = np.array([it["mejor"] for it in train])
    scaler = StandardScaler().fit(Xtr)
    gate = LogisticRegression(max_iter=2000, random_state=args.seed)
    gate.fit(scaler.transform(Xtr), ytr)
    acc_gate = float(gate.score(scaler.transform(
        np.array([it["feats"] for it in test])),
        np.array([it["mejor"] for it in test])))

    # Estrategias sobre el test.
    def pick_gate(it):
        p = gate.predict_proba(scaler.transform([it["feats"]]))[0]
        return it["pb"] if p[1] >= p[0] else it["pa"]

    res = {
        "n_tiles": len(items), "n_train": len(train), "n_test": len(test),
        "experto_a": {"ckpt": args.experto_a, "iou_medio_test":
                      _iou_estrategia(test, lambda it: it["pa"])},
        "experto_b": {"ckpt": args.experto_b, "iou_medio_test":
                      _iou_estrategia(test, lambda it: it["pb"])},
        "promedio": {"iou_medio_test":
                     _iou_estrategia(test, lambda it: (it["pa"] + it["pb"]) / 2)},
        "compuerta": {"iou_medio_test": _iou_estrategia(test, pick_gate),
                      "acierto_eleccion": round(acc_gate, 4)},
        "oraculo": {"iou_medio_test":
                    float(np.mean([max(it["iou_a"], it["iou_b"]) for it in test]))},
        "veredicto": "",
    }
    a = res["experto_a"]["iou_medio_test"]
    b = res["experto_b"]["iou_medio_test"]
    g = res["compuerta"]["iou_medio_test"]
    mejor_fijo = max(a, b)
    if g > mejor_fijo + 0.005:
        res["veredicto"] = (f"La compuerta SUPERA al mejor fijo "
                            f"({g:.4f} vs {mejor_fijo:.4f})")
    else:
        res["veredicto"] = (f"La compuerta NO supera al mejor fijo "
                            f"({g:.4f} vs {mejor_fijo:.4f}); con estos dos "
                            f"expertos el MoE no aporta")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    # Artefacto del gate para la integración en post-vuelo (cansat/moe.py).
    gate_out = Path(args.guardar_gate)
    gate_out.parent.mkdir(parents=True, exist_ok=True)
    gate_out.write_text(json.dumps({
        "features": list(FEATURES),
        "scaler": {"mean": [float(v) for v in scaler.mean_],
                   "scale": [float(v) for v in scaler.scale_]},
        "coef": [float(v) for v in gate.coef_[0]],
        "intercept": float(gate.intercept_[0]),
        "meta": {
            "generado": time.strftime("%Y-%m-%d"),
            "n_tiles": len(items), "n_train": len(train), "n_test": len(test),
            "seed": args.seed, "acierto_eleccion": round(acc_gate, 4),
            "experto_a": args.experto_a, "experto_b": args.experto_b,
            "iou_compuerta": round(g, 4), "iou_mejor_fijo": round(mejor_fijo, 4),
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → gate: {gate_out}")
    print("=" * 64)
    print(f"  A (xBD) solo      : IoU {a:.4f}")
    print(f"  B (two-stage UAV) : IoU {b:.4f}")
    print(f"  Promedio          : IoU {res['promedio']['iou_medio_test']:.4f}")
    print(f"  Compuerta         : IoU {g:.4f} (acierto de elección {acc_gate:.1%})")
    print(f"  Oráculo           : IoU {res['oraculo']['iou_medio_test']:.4f}")
    print(f"  → {res['veredicto']}")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
