"""
CanSat La Base — Mapa de corredor del terreno (entregable DPD).

Apila los frames de evidencia ordenados por altitud descendente en una tira
vertical: es el "corredor" que recorrió el CanSat desde la eyección hasta el
suelo.

ARREGLADO:
  · ``cv2.resize(img, (width, int(h * width / w)))`` podía devolver alto 0 con
    un ``width`` chico y una evidencia muy apaisada → ``vconcat`` reventaba.
    Ahora el alto se clampea a >= 1.
  · ``open()`` sin ``with``.
  · ``rows[:args.max]`` cortaba ANTES de descartar las evidencias que no
    existían, así que el mapa podía quedar con menos frames de los pedidos aun
    habiendo más disponibles. Ahora se filtran primero y se corta después.
  · Si todas las tiles quedan de distinto ancho por un redondeo, ``vconcat``
    falla: se normaliza el ancho explícitamente.
  · ``--csv`` ahora acepta cualquier telemetría (la versionada del pipeline
    v7 incluye sufijo horario).

Uso:
    python corridor_map.py
    python corridor_map.py --csv outputs/mission/telemetry_20260916_1200.csv --max 16
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
from cansat import paths as PROJ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mapa de corredor CanSat")
    ap.add_argument("--csv", default=str(PROJ.TELEMETRY_CSV))
    ap.add_argument("--vis", default=str(PROJ.MISSION_DIR / "vis"))
    ap.add_argument("--suffix", default="_evid",
                    help="sufijo del archivo de evidencia (default: _evid)")
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--out", default=str(PROJ.CORRIDOR_MAP))
    args = ap.parse_args(argv)

    src = Path(args.csv)
    if not src.is_file():
        print(f"[ERROR] No existe {src}\n  → Corré primero mission_pipeline.py")
        return 1

    with src.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"[ERROR] {src} no tiene filas.")
        return 1

    # Orden por altitud descendente: apogeo → suelo.
    def alt_of(r: dict) -> float:
        try:
            return float(r.get("alt_m") or 0.0)
        except ValueError:
            return 0.0

    rows.sort(key=alt_of, reverse=True)

    vis_dir = Path(args.vis)
    tiles: list[np.ndarray] = []
    faltan: list[str] = []
    for r in rows:
        name = str(r.get("src") or "")
        if not name:
            continue
        p = vis_dir / f"{name}{args.suffix}.jpg"
        img = cv2.imread(str(p)) if p.is_file() else None
        if img is None:
            # Probar otras extensiones antes de darla por perdida.
            for ext in (".png", ".jpeg", ".webp"):
                alt_p = vis_dir / f"{name}{args.suffix}{ext}"
                if alt_p.is_file():
                    img = cv2.imread(str(alt_p))
                    break
        if img is None:
            faltan.append(name)
            continue

        h, w = img.shape[:2]
        # ⚠ Antes: int(h * width / w) podía dar 0 → vconcat reventaba.
        new_h = max(1, round(h * args.width / max(1, w)))
        tile = cv2.resize(img, (args.width, new_h))
        # vconcat exige el mismo ancho en todas: forzamos.
        if tile.shape[1] != args.width:
            tile = cv2.resize(tile, (args.width, max(1, tile.shape[0])))
        tiles.append(tile)
        if len(tiles) >= args.max:
            break

    if not tiles:
        print("[ERROR] No encontré ninguna evidencia.")
        print(f"        Busqué en {vis_dir} con sufijo '{args.suffix}'.")
        print("        → Corré antes mission_pipeline.py sin --no-vis.")
        return 1

    header = np.zeros((60, args.width, 3), dtype=np.uint8)
    cv2.putText(header, "MAPA DE CORREDOR  (apogeo -> suelo)",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    final = cv2.vconcat([header, *tiles])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), final, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[OK] {len(tiles)} frames apilados → {out} "
          f"({final.shape[1]}x{final.shape[0]})")
    if faltan:
        print(f"     {len(faltan)} frame(s) sin evidencia: "
              f"{', '.join(faltan[:6])}{'…' if len(faltan) > 6 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
