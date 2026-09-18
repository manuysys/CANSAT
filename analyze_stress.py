"""
CanSat La Base — Análisis de estrés ambiental sobre máscaras de segmentación.

Los índices y el veredicto viven en ``cansat/indices.py`` (FUENTE ÚNICA). Este
archivo queda como:
  · CLI de inspección rápida sobre máscaras coloreadas ya generadas, y
  · el conversor máscara-JPG → mapa de clases (``mask_to_classes``).

⚠ ``mask_to_classes`` reconstruye las clases a partir de una imagen **JPG
  coloreada**, asignando a cada píxel el color de paleta más cercano. Es
  inherentemente lossy y sólo tiene sentido para inspeccionar salida visual ya
  escrita en disco. El pipeline de misión pasa el ``seg_map`` directamente y
  **no debe** usar esta ruta.

Uso:
    python analyze_stress.py --image outputs/test_inference/Rural/4472_mask.jpg
    python analyze_stress.py --folder outputs/test_inference/Rural
    python analyze_stress.py --pcts 30,12,5,40,13      # índices sin imagen
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
from cansat import indices as IDX

# Reexport por compatibilidad con quien importaba de acá.
CLASS_NAMES = list(IDX.CLASS_NAMES)

# Colores BGR usados al generar la máscara. DEBEN coincidir con
# inference.COLOR_PALETTE y post_flight.COLORS — por eso están definidos una
# sola vez, en inference.py, y se importan.
try:
    from inference import COLOR_PALETTE
    CLASS_COLORS_BGR = [COLOR_PALETTE[i] for i in range(len(CLASS_NAMES))]
except Exception:
    CLASS_COLORS_BGR = [
        (0, 200, 0), (180, 100, 0), (255, 100, 0),
        (0, 200, 255), (128, 128, 128),
    ]

MIN_PATCH_PX = 100   # parches de verde más chicos se ignoran (ruido)


def mask_to_classes(mask_bgr: np.ndarray) -> np.ndarray:
    """
    Convierte una máscara coloreada (BGR) en un mapa de clases 0..4.

    Asigna a cada píxel el color de paleta MÁS CERCANO (no igualdad exacta),
    así tolera que el JPG haya movido un poco los colores.
    """
    h, w = mask_bgr.shape[:2]
    p = mask_bgr.reshape(-1, 3).astype(np.float32)         # (N, 3)
    pal = np.array(CLASS_COLORS_BGR, dtype=np.float32)     # (5, 3)

    # Distancia al cuadrado píxel→color:  |p|² − 2·p·c + |c|²
    pp = (p * p).sum(axis=1)[:, None]        # (N, 1)
    pc = p @ pal.T                           # (N, 5)
    cc = (pal * pal).sum(axis=1)[None, :]    # (1, 5)
    return (pp - 2 * pc + cc).argmin(axis=1).reshape(h, w).astype(np.int8)


def green_patches(seg_map: np.ndarray, valid: np.ndarray | None = None,
                  min_px: int = MIN_PATCH_PX) -> int:
    """Cantidad de parches de vegetación de al menos ``min_px`` píxeles."""
    veg = (seg_map == 0) if valid is None else ((seg_map == 0) & valid)
    veg = veg.astype(np.uint8)
    if not veg.any():
        return 0
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(veg, connectivity=8)
    if n <= 1:
        return 0
    return int((stats[1:, cv2.CC_STAT_AREA] >= min_px).sum())


def analyze(seg_map: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """
    Calcula índices ambientales sobre un mapa de clases 0..4.

    Mantiene la firma y las claves históricas (``pcts``, ``usi``, ``ndvi``,
    ``density``, ``n_green_patches``, ``frag_per_patch``, ``flood_risk``,
    ``verdict``) para no romper consumidores, y agrega las canónicas
    (``gvi``, ``usi_norm``, ``valid_frac``, ``vcode``).

    ``valid``: máscara bool (True = píxel con datos). Si es None, usa todo.
    """
    seg_map = np.asarray(seg_map)
    if valid is None:
        valid = np.ones(seg_map.shape, dtype=bool)
    valid = np.asarray(valid, dtype=bool)

    total = max(1, int(valid.sum()))
    pcts = [float(((seg_map == c) & valid).sum()) / total * 100.0
            for c in range(len(CLASS_NAMES))]

    env = IDX.environment(pcts,
                          n_green_patches=green_patches(seg_map, valid),
                          valid_frac=float(valid.mean()))
    env["pcts"] = dict(zip(CLASS_NAMES, pcts, strict=False))
    return env


def print_report(mask_path, r: dict) -> None:
    print("\n" + "=" * 62)
    print("  ANÁLISIS DE ESTRÉS AMBIENTAL")
    print(f"  Imagen: {getattr(mask_path, 'name', mask_path)}")
    print("=" * 62)

    print("\n  Cobertura del terreno:")
    for name, v in r["pcts"].items():
        print(f"    {name:<14} {v:5.1f}%  {'█' * int(v // 4)}")

    vf = r.get("valid_frac", 1.0)
    if vf < 1.0:
        print(f"\n    píxeles analizados : {vf * 100:.1f}% "
              f"({(1 - vf) * 100:.1f}% sin datos, excluidos)")

    print("\n  Índices ambientales:")
    print(f"    USI (estrés urbano)  : {r['usi']:.2f}   ← cociente, NO acotado a [0,1]")
    print(f"    USI normalizado      : {r['usi_norm']:.3f}  ← este sí va de 0 a 1")
    print(f"    GVI (verdor relativo): {r['gvi']:+.3f}  "
          f"({'saludable' if r['gvi'] > 0 else 'degradado'})")
    print("      ⚠ NO es NDVI: no hay banda NIR. Ver cansat/indices.py.")
    print(f"    Densidad urbana      : {r['density']:.1f}% del área no vegetal")
    print(f"    Parches de verde     : {r['n_green_patches']} (>= {MIN_PATCH_PX} px)")
    print(f"    Verde por parche     : {r['frag_per_patch']:.2f}%")
    print(f"    Riesgo hídrico       : {r['flood_risk']:.2f}")
    print(f"\n  ▸ VEREDICTO: {r['verdict']}  (código {r['vcode']})")
    print("=" * 62)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Análisis de estrés ambiental sobre máscaras o porcentajes")
    ap.add_argument("--image", default=None, help="una máscara coloreada (*_mask.jpg)")
    ap.add_argument("--folder", default=None, help="carpeta con *_mask.jpg")
    ap.add_argument("--pcts", default=None,
                    help="percentages directos: veg,bui,wat,bare,oth (sin imagen)")
    args = ap.parse_args(argv)

    if args.pcts:
        try:
            pcts = [float(x) for x in args.pcts.split(",")]
        except ValueError:
            print("[ERROR] --pcts espera 5 números separados por coma.")
            return 1
        if len(pcts) != len(CLASS_NAMES):
            print(f"[ERROR] --pcts espera {len(CLASS_NAMES)} valores, llegaron {len(pcts)}.")
            return 1
        env = IDX.environment(IDX.normalize_pcts(pcts))
        env["pcts"] = dict(zip(CLASS_NAMES, pcts, strict=False))
        print_report(f"pcts={args.pcts}", env)
        return 0

    if args.image:
        paths = [Path(args.image)]
    elif args.folder:
        paths = sorted(Path(args.folder).glob("*_mask.jpg"))
    else:
        print("[ERROR] Especificá --image, --folder o --pcts")
        return 1

    if not paths:
        print("[ERROR] No se encontraron máscaras.")
        return 1

    for p in paths:
        mask = cv2.imread(str(p))
        if mask is None:
            print(f"[WARN] No se pudo leer: {p}")
            continue
        print_report(p, analyze(mask_to_classes(mask)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
