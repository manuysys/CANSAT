"""
Calibración por clase del clasificador de tipo de desastre.

Ajusta una temperatura por clase (``q_i ∝ p_i^(1/T_i)``) sobre eventos de
calibración y la evalúa en eventos held-out, comparando contra la T global.
Se adopta solo si baja el ECE de evaluación Y preserva el ranking de
p_incendio (la política ``fire_only_v1`` decide por umbral sobre ella).

Uso:
    python tools/calibrate_per_class.py
    python tools/calibrate_per_class.py --probs outputs/loeo_probs_todas.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cansat.calibracion import analizar_por_clase, leer_probs  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibración por clase (tipo)")
    ap.add_argument("--probs", default="outputs/loeo_probs_todas.csv")
    ap.add_argument("--out", default="docs/benchmarks/calibracion_por_clase.json")
    args = ap.parse_args(argv)

    p = Path(args.probs)
    if not p.is_file():
        print(f"[ERROR] no existe {p}. Generá el CSV con:\n"
              f"  python train_disaster_type.py --loeo --probs-todas {p}")
        return 1
    eventos, probs, y = leer_probs(p)
    if len(y) == 0:
        print("[ERROR] CSV vacío")
        return 1
    print(f"  {len(y)} tiles · {len(set(eventos))} eventos · "
          f"{len(set(eventos)) // 2} para calibrar")
    res = analizar_por_clase(probs, y, eventos)
    res["generado"] = datetime.now(timezone.utc).isoformat()
    res["script"] = "tools/calibrate_per_class.py"
    res["fuente"] = str(p)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"  ECE evaluación: crudo {res['ece_evaluacion_crudo']:.4f} · "
          f"global {res['ece_evaluacion_global']:.4f} · "
          f"por-clase {res['ece_evaluacion_por_clase']:.4f}")
    print(f"  NLL evaluación: crudo {res['nll_evaluacion_crudo']:.4f} · "
          f"global {res['nll_evaluacion_global']:.4f} · "
          f"por-clase {res['nll_evaluacion_por_clase']:.4f}")
    print("  Temperaturas por clase: " + ", ".join(
        f"{k}={v['temperatura']} (n={v['n_calibracion']})"
        for k, v in res["temperaturas_por_clase"].items()))
    print(f"  Ranking incendio preservado: {res['ranking_incendio_preservado']}")
    if res["adoptada"]:
        print("  [OK] la T por clase mejora y preserva el ranking: ADOPTADA")
    else:
        print("  [WARN] no mejora o rompe el ranking: NO se adopta "
              "(se documenta igual)")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
