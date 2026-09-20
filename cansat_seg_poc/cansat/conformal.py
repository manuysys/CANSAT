"""
conformal.py - Umbral de abstención con cobertura estadística (split conformal).

Reemplazo riguroso del umbral 0.70 "provisional" de fire_only_v1: en vez de
elegir un número a ojo, se CALIBRA sobre scores de eventos held-out (LOEO) y se
controla la tasa de falsos positivos de incendio con validez de muestra finita.

Construcción (binaria, un solo umbral):
  · scores de calibración = p_incendio de tiles NO-incendio (eventos held-out
    del LOEO);
  · umbral τ(α) = cuantil ⌈(n+1)(1-α)⌉ de esos scores → se confirma incendio
    cuando p_incendio >= τ, con FPR ≤ α y probabilidad ≥ 1-α (split conformal,
    Vovk; aplicado a RS p.ej. en ml-for-rs). Si n es chico, el cuantil se
    satura: se reporta n y α efectivo.
  · el RECALL de incendio NO se garantiza: se MIDE sobre los eventos de
    incendio held-out y se reporta junto al umbral (no se inventa).

Uso:
    python -m cansat.conformal --probs outputs/loeo_probs.csv --alpha 0.05
(csv con columnas: evento, es_incendio, p_incendio)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def cuantil_conformal(scores_no_fuego, alpha: float = 0.05) -> float:
    """τ tal que FPR ≤ alpha con validez de muestra finita (split conformal).

    ``scores_no_fuego`` son los scores (1 - p) de los tiles NO-incendio de
    calibración. Devuelve 1.0 si no hay calibración suficiente (abstiene todo).
    """
    s = sorted(float(x) for x in scores_no_fuego)
    n = len(s)
    if n == 0 or not 0.0 < alpha < 1.0:
        return 1.0
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return 1.0  # sin datos para ese alpha: abstención total (fail-safe)
    return s[k - 1]


def recall(scores_incendio, umbral: float) -> tuple[float, int, int]:
    """Recall medido: (tasa, confirmados, total) de tiles de incendio.

    Convención: score = p_incendio; se confirma con score >= umbral.
    """
    s = [float(x) for x in scores_incendio]
    if not s:
        return 0.0, 0, 0
    ok = sum(1 for x in s if x >= umbral)
    return ok / len(s), ok, len(s)


def fpr(scores_no_fuego, umbral: float) -> tuple[float, int, int]:
    """FPR medido en calibración (informativo; la garantía es el cuantil)."""
    s = [float(x) for x in scores_no_fuego]
    if not s:
        return 0.0, 0, 0
    fp = sum(1 for x in s if x >= umbral)
    return fp / len(s), fp, len(s)


def evaluar(probs_csv: Path, alphas=(0.01, 0.05, 0.10, 0.20)) -> dict:
    """Calibra τ por alpha y reporta recall/FPR medidos por evento."""
    no_fuego: list[float] = []
    fuego: list[float] = []
    por_evento: dict[str, list[float]] = {}
    with Path(probs_csv).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            score = float(r["p_incendio"])
            es = str(r["es_incendio"]).strip().lower() in ("1", "true", "si", "sí")
            (fuego if es else no_fuego).append(score)
            por_evento.setdefault(r["evento"], []).append(score)
    filas = []
    for a in alphas:
        tau = cuantil_conformal(no_fuego, a)
        rec, ok, tot = recall(fuego, tau)
        fp, nfp, ncal = fpr(no_fuego, tau)
        filas.append({"alpha": a, "umbral_tau": round(tau, 4),
                      "recall_incendio": round(rec, 4),
                      "confirmados": ok, "tiles_incendio": tot,
                      "fpr_calibracion": round(fp, 4),
                      "fp": nfp, "n_calibracion": ncal})
    return {"n_incendio": len(fuego), "n_no_incendio": len(no_fuego),
            "eventos": sorted(por_evento), "filas": filas,
            "nota": ("τ garantiza FPR<=alpha en muestra finita; el recall es "
                     "MEDIDO en eventos de incendio held-out, no garantizado. "
                     "Con 2 eventos de incendio la incertidumbre del recall es "
                     "alta: no extrapolar.")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probs", default="outputs/loeo_probs.csv")
    ap.add_argument("--json-out", default="outputs/conformal_incendio.json")
    args = ap.parse_args()
    res = evaluar(Path(args.probs))
    out = Path(args.json_out)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    for f in res["filas"]:
        print(f"  alpha {f['alpha']:.2f} -> tau {f['umbral_tau']:.3f} | "
              f"recall {f['recall_incendio']:.3f} "
              f"({f['confirmados']}/{f['tiles_incendio']}) | "
              f"FPR calib {f['fpr_calibracion']:.3f} "
              f"({f['fp']}/{f['n_calibracion']})")
    print(f"[OK] {out} (n_incendio={res['n_incendio']}, "
          f"n_no_incendio={res['n_no_incendio']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
