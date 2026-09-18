"""
CanSat La Base — Quality gate de nitidez (Laplaciano).

Lee ``telemetry.csv``, separa los frames nítidos de los borrosos y escribe
``telemetry_clean.csv``.

El pipeline de vuelo ya **marca** los frames bajo el umbral (``--sharp-gate``) y
los degrada a thumbnail en el sampler, pero no los borra del CSV: en vuelo es
mejor conservar todo y filtrar en tierra, donde se puede revisar. Este script
es ese filtro de tierra.

ARREGLADO respecto de la versión anterior:
  · ``rows[0]`` lanzaba ``IndexError`` con un CSV vacío (caso real: un vuelo que
    no llegó a escribir nada).
  · ``open()`` sin ``with`` ni ``close``.
  · Una fila con ``sharp`` vacío o no numérico mataba todo el archivo con
    ``ValueError``; ahora se reporta y se descarta sola.
  · ``--report`` permite sólo mirar sin escribir.

Uso:
    python quality_gate.py
    python quality_gate.py --csv outputs/mission/telemetry.csv --thresh 50
    python quality_gate.py --report
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cansat import paths as PROJ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Quality gate de nitidez")
    ap.add_argument("--csv", default=str(PROJ.TELEMETRY_CSV))
    ap.add_argument("--thresh", type=float, default=50.0,
                    help="Laplaciano mínimo (el bench v2 dio 50-80 para frames nítidos)")
    ap.add_argument("--out", default=None,
                    help="CSV de salida (default: telemetry_clean.csv junto al de entrada)")
    ap.add_argument("--report", action="store_true",
                    help="sólo imprimir el resumen, sin escribir nada")
    args = ap.parse_args(argv)

    src = Path(args.csv)
    if not src.is_file():
        print(f"[ERROR] No existe {src}\n  → Corré primero mission_pipeline.py")
        return 1

    with src.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        print(f"[ERROR] {src} no tiene cabecera.")
        return 1
    if not rows:
        # Antes: IndexError en rows[0].keys(). Caso real de un vuelo abortado.
        print(f"[WARN] {src} tiene cabecera pero 0 filas. Nada que filtrar.")
        return 0
    if "sharp" not in fieldnames:
        print(f"[ERROR] {src} no tiene la columna 'sharp'.\n"
              f"        Columnas presentes: {fieldnames}\n"
              f"  → Es un CSV de una versión vieja del pipeline; regeneralo.")
        return 1

    ok, borrosos, invalidos = [], [], []
    for r in rows:
        raw = (r.get("sharp") or "").strip()
        try:
            s = float(raw)
        except ValueError:
            invalidos.append((r.get("src", "?"), raw))
            continue
        (ok if s >= args.thresh else borrosos).append((r.get("src", "?"), s))

    total = len(rows)
    print("=" * 62)
    print("  QUALITY GATE — nitidez (varianza del Laplaciano)")
    print("=" * 62)
    print(f"  CSV        : {src}")
    print(f"  Umbral     : sharp >= {args.thresh}")
    print(f"  Total      : {total}")
    print(f"  Aceptados  : {len(ok)}")
    print(f"  Descartados: {len(borrosos)}")
    if invalidos:
        print(f"  Sin dato   : {len(invalidos)}  ← filas con 'sharp' vacío o inválido")
    if ok:
        vals = [s for _, s in ok]
        print(f"  Sharp de los aceptados: min {min(vals):.0f} · "
              f"mediana {sorted(vals)[len(vals) // 2]:.0f} · max {max(vals):.0f}")
    if borrosos:
        print("\n  Frames descartados:")
        for src_name, s in sorted(borrosos, key=lambda x: x[1])[:20]:
            print(f"    {src_name:<20} sharp {s:8.1f}")
        if len(borrosos) > 20:
            print(f"    … y {len(borrosos) - 20} más")
    if not ok:
        print("\n  ⚠ NINGÚN frame supera el umbral. Antes de descartar todo el")
        print("    vuelo, revisá si el umbral tiene sentido para estas imágenes:")
        print("      python quality_gate.py --report --thresh 10")
        print("    La varianza del Laplaciano depende fuerte de la resolución y")
        print("    del contenido de la escena, así que 50 no es universal.")

    if args.report:
        return 0

    dst = Path(args.out) if args.out else src.with_name("telemetry_clean.csv")
    with dst.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        keep = {name for name, _ in ok}
        w.writerows(r for r in rows if r.get("src") in keep)
    print(f"\n  → {dst}  ({len(ok)} filas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
