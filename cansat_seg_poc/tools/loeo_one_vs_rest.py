"""
Multiclase vs one-vs-rest (binario por clase) para el tipo de desastre — LOEO.

Hipótesis a testear (pedido del equipo): "¿un modelo por tipo de desastre
generaliza mejor que el clasificador multiclase de 7 clases?".

Diseño JUSTO:
  · mismo split leave-one-event-out, misma semilla y mismos tiles que el LOEO
    canónico (``train_disaster_type.py --loeo``);
  · **mismo presupuesto** para las dos ramas (``--epochs`` y ``--max-por-clase``
    reducidos, porque la rama OvR entrena 7× más modelos);
  · métrica principal: **Average Precision por clase** (sin umbral) y su macro;
    también se reporta el weighted (ponderado por frecuencia).
  · veredicto: se adopta solo si el AP macro OvR supera al del control.

Salida: JSON incremental en ``docs/benchmarks/tipo_one_vs_rest.json``.

Uso:
    python tools/loeo_one_vs_rest.py --epochs 3 --max-por-clase 300
    python tools/loeo_one_vs_rest.py --eventos xbd:woolsey-fire --clases incendio --epochs 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from torch.utils.data import DataLoader                       # noqa: E402

from cansat.seed import set_seed                              # noqa: E402
from train_disaster_type import (                             # noqa: E402
    CLASES,
    TipoDS,
    _entrenar_fold,
    modelo,
    recolectar_eventos,
)


def ap_por_clase(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Average Precision por clase (área bajo la curva precision-recall)."""
    aps: dict[str, float] = {}
    for c, nombre in enumerate(CLASES):
        pos = y == c
        n_pos = int(pos.sum())
        if n_pos == 0:
            continue
        orden = np.argsort(-probs[:, c], kind="stable")
        y_ord = pos[orden]
        tp = np.cumsum(y_ord)
        fp = np.cumsum(~y_ord)
        precision = tp / np.maximum(tp + fp, 1)
        aps[nombre] = round(float((precision * y_ord).sum() / n_pos), 4)
    return aps


def entrenar_binario(train, test, args, device, clase: int) -> np.ndarray:
    """Entrena ``y == clase`` vs resto y devuelve p(clase) en test."""
    model = modelo(2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dl = DataLoader(TipoDS(train, args.size, aug=True), batch_size=args.batch,
                    shuffle=True, num_workers=4, persistent_workers=True)
    for _ in range(args.epochs):
        model.train()
        for x, y in dl:
            x = x.to(device)
            yb = (y.to(device) == clase).long()
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(x), yb).backward()
            opt.step()
    model.eval()
    te_dl = DataLoader(TipoDS(test, args.size), batch_size=args.batch, num_workers=4)
    out: list[float] = []
    with torch.no_grad():
        for x, _y in te_dl:
            pr = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            out.extend(pr[:, 1].tolist())
    return np.asarray(out, dtype=np.float64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LOEO multiclase vs one-vs-rest")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-por-clase", type=int, default=300)
    ap.add_argument("--loss", choices=("ce", "balanceada", "focal"), default="ce",
                    help="pérdida para AMBAS ramas (default: CE, el canónico)")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--eventos", default="", help="subset de eventos (coma) para humo")
    ap.add_argument("--clases", default="", help="subset de clases OvR (coma) para humo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="docs/benchmarks/tipo_one_vs_rest.json")
    args = ap.parse_args(argv)
    set_seed(args.seed)

    filas = recolectar_eventos(args.max_por_clase, args.seed)
    eventos = sorted({e for _p, _c, e in filas})
    if args.eventos:
        elegidos = {e.strip() for e in args.eventos.split(",") if e.strip()}
        eventos = [e for e in eventos if e in elegidos]
    clases_ovr = list(range(len(CLASES)))
    if args.clases:
        pedidas = {c.strip() for c in args.clases.split(",") if c.strip()}
        clases_ovr = [i for i, c in enumerate(CLASES) if c in pedidas]
    print(f"LOEO {len(eventos)} eventos · {len(filas)} tiles · "
          f"epochs {args.epochs} · max/clase {args.max_por_clase} · "
          f"OvR de {[CLASES[c] for c in clases_ovr]}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "script": "tools/loeo_one_vs_rest.py",
        "protocolo": {
            "split": "leave-one-event-out (mismo que el canonico)",
            "epochs": args.epochs,
            "max_por_clase": args.max_por_clase,
            "seed": args.seed,
            "clases_ovr": [CLASES[c] for c in clases_ovr],
            "nota": ("control y OvR con el MISMO presupuesto reducido; "
                     "comparacion por AP por clase (sin umbral)"),
        },
        "eventos": {},
        "completo": False,
    }
    control_probs: list[np.ndarray] = []
    control_y: list[np.ndarray] = []
    ovr_probs: list[np.ndarray] = []
    t0 = time.time()
    for ev in eventos:
        test = [(p, c) for p, c, e in filas if e == ev]
        train = [(p, c) for p, c, e in filas if e != ev]
        if len(test) < 30 or len(train) < 200:
            print(f"[skip] {ev}: test {len(test)} / train {len(train)}", flush=True)
            continue
        y_test = np.asarray([c for _p, c in test], dtype=np.int64)

        # Rama control (multiclase, mismo presupuesto).
        _acc, _pr, _re, probs = _entrenar_fold(train, test, args, device,
                                               epochs=args.epochs)
        p_ctrl = np.asarray([v for _y, v in probs], dtype=np.float64)
        control_probs.append(p_ctrl)
        control_y.append(y_test)

        # Rama one-vs-rest: un binario por clase pedida.
        p_ovr = np.full((len(test), len(CLASES)), np.nan)
        for c in clases_ovr:
            p_ovr[:, c] = entrenar_binario(train, test, args, device, c)
        ovr_probs.append(p_ovr)

        ev_res = {
            "n_test": len(test),
            "ap_control": ap_por_clase(p_ctrl, y_test),
            "ap_ovr": ap_por_clase(np.nan_to_num(p_ovr, nan=0.0), y_test),
        }
        payload["eventos"][ev] = ev_res
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"  {ev}: n={len(test)} · AP ctrl {ev_res['ap_control']} · "
              f"AP ovr {ev_res['ap_ovr']}", flush=True)

    if control_probs:
        y_all = np.concatenate(control_y)
        p_ctrl_all = np.vstack(control_probs)
        p_ovr_all = np.vstack(ovr_probs)
        ap_ctrl = ap_por_clase(p_ctrl_all, y_all)
        ap_ovr = ap_por_clase(np.nan_to_num(p_ovr_all, nan=0.0), y_all)
        comunes = [c for c in ap_ctrl if c in ap_ovr]
        macro_ctrl = round(float(np.mean([ap_ctrl[c] for c in comunes])), 4)
        macro_ovr = round(float(np.mean([ap_ovr[c] for c in comunes])), 4)
        cuenta = {CLASES[c]: int((y_all == c).sum()) for c in range(len(CLASES))}
        w_ctrl = round(float(np.average([ap_ctrl[c] for c in comunes],
                                        weights=[cuenta[c] for c in comunes])), 4)
        w_ovr = round(float(np.average([ap_ovr[c] for c in comunes],
                                       weights=[cuenta[c] for c in comunes])), 4)
        adoptada = bool(macro_ovr > macro_ctrl + 0.02)   # margen 2 pts de AP
        payload.update({
            "completo": True,
            "n_tiles_evaluados": len(y_all),
            "clases_evaluadas": comunes,
            "ap_control": ap_ctrl,
            "ap_ovr": ap_ovr,
            "macro_ap_control": macro_ctrl,
            "macro_ap_ovr": macro_ovr,
            "weighted_ap_control": w_ctrl,
            "weighted_ap_ovr": w_ovr,
            "adoptada": adoptada,
            "minutos": round((time.time() - t0) / 60.0, 1),
            "veredicto": ("se adopta OvR si macro AP mejora >2 pts; si no, "
                          "se documenta como no adoptada"),
        })
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"[OK] macro AP control {macro_ctrl} vs OvR {macro_ovr} · "
              f"weighted {w_ctrl} vs {w_ovr} · adoptada={adoptada} "
              f"({payload['minutos']} min)", flush=True)
    print(f"  → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
