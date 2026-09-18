"""
Validación INT8 sobre **métricas de misión** (veredictos, alertas, terreno %).

Compara el ONNX FP32 contra su versión INT8 sobre los frames de un vuelo o
simulacro, y decide si el INT8 es apto para la Raspberry Pi.

════════════════════════════════════════════════════════════════════════════
POR QUÉ SE REESCRIBIÓ COMPLETO
════════════════════════════════════════════════════════════════════════════
La decisión registrada en ``docs/reporte_pruebas.md`` —

    "Cuantización INT8 descartada para vuelo: coincidencia 87.6% < umbral 95%;
     se vuela con FP32."

— salía de la versión anterior de este script, que **no medía lo que decía
medir**:

 1. **Alimentaba al modelo con las evidencias anotadas.** Buscaba
    ``outputs/mission/vis/*_evidence.jpg``, que no existe (el pipeline escribe
    ``*_evid.jpg``), y caía al ``glob("*.jpg")[:12]``: frames con el overlay de
    segmentación al 40 %, los recuadros de YOLO y la banda negra de texto
    **quemados encima**. Ningún modelo fue entrenado con eso. Es la causa más
    probable del 87.6 % bajo, y no tiene nada que ver con la cuantización.
 2. **Usaba USI/NDVI/veredictos distintos** a los del pipeline de vuelo
    (``usi = (bui+bare+0.5*oth)/max(veg,1e-3)``, ``ndvi = (veg-wat)/(veg+wat)``,
    umbrales 1.0/30.0 en vez de 1.0/3.0). Medía otra métrica de misión.
 3. **Comparaba modelos distintos**: FP32 = ``best_terrain_cbam.pth`` por
    PyTorch, INT8 = ``cansat_seg_terrain_v2_int8.onnx``. No era "el mismo
    modelo cuantizado".
 4. **``torch.device("cuda")`` hardcodeado**, sin fallback: imposible de correr
    en una máquina sin NVIDIA (por eso nunca se pudo reproducir).
 5. **Porcentajes de terreno como media de probabilidades** (``probs[c].mean()``)
    en vez del argmax por píxel que usa el pipeline.
 6. **División por cero** si no encontraba frames (``n = len(frames)``).
 7. **No excluía los píxeles sin datos**, así que el borde negro de los tiles
    entraba en la comparación.

Ahora compara **el mismo grafo ONNX en FP32 y en INT8**, sobre **frames crudos**
(``high_res``/``full_res``, o la carpeta que se le pase), con los índices de
``cansat.indices`` —los mismos que vuelan—, argmax por píxel, máscara de
no-data, device con fallback, y acuerdo de veredicto / alerta / terreno medidos
por separado.

Uso:
    python validate_int8_mission.py
    python validate_int8_mission.py --frames dataset/loveda_raw/Test/Urban/images_png --n 40
    python validate_int8_mission.py --fp32 outputs/cansat_seg_terrain_v2.onnx \
                                    --int8 outputs/cansat_seg_terrain_v2_int8.onnx
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
from cansat import indices as IDX
from cansat import nodata as ND
from cansat import onnxio
from cansat import paths as PROJ
from cansat import preprocess as PP

# Umbrales de decisión. Los publica MODELS.yaml y el informe.
UMBRAL_ARGMAX = 0.95     # acuerdo píxel a píxel exigido
UMBRAL_VEREDICTO = 0.95  # acuerdo de veredicto exigido
UMBRAL_ALERTA = 1.00     # las alertas no pueden divergir


def _muestrear(cands: list[Path], n: int, seed: int) -> list[Path]:
    """
    Muestreo reproducible y sin frames sintéticos.

    Antes era ``cands[:n]``: los primeros N por orden alfabético, así que la
    muestra dependía de los nombres y podía quedar dominada por frames de
    ``simulate_disaster.py`` (``*_simulado*``), que no son de vuelo.
    """
    reales = [p for p in cands
              if "simulado" not in p.stem.lower()
              and not p.stem.lower().startswith(("sim_", "demo_"))]
    if len(reales) >= min(3, n):
        cands = reales
    rng = random.Random(seed)
    if n >= len(cands):
        return sorted(cands)
    return sorted(rng.sample(cands, n))


def collect_frames(args) -> list[Path]:
    """
    Frames **crudos** para validar. Nunca las evidencias anotadas.

    Orden de preferencia:
      1. ``--frames`` explícito.
      2. ``outputs/mission/high_res`` y ``full_res`` (lo que guardó el sampler).
      3. La carpeta de origen registrada en ``telemetry.csv`` no es confiable
         (puede ser un dataset de entrenamiento), así que no se usa.
    """
    if args.frames:
        d = Path(args.frames)
        if not d.is_dir():
            raise FileNotFoundError(f"No existe la carpeta de frames: {d}")
        cands = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        if not cands:
            raise FileNotFoundError(f"No hay imágenes en {d}")
        return _muestrear(cands, args.n, args.seed)

    for sub in ("high_res", "full_res", "thumb"):
        d = PROJ.MISSION_DIR / sub
        if d.is_dir():
            cands = sorted(p for p in d.iterdir()
                           if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
            if cands:
                print(f"  Usando frames crudos de {d} ({len(cands)} disponibles)")
                return _muestrear(cands, args.n, args.seed)

    raise FileNotFoundError(
        "No encontré frames crudos para validar.\n"
        "  · Pasá una carpeta con --frames, o\n"
        "  · corré antes mission_pipeline.py --sampler (genera high_res/full_res).\n"
        "  ⚠ NO se usan outputs/mission/vis/*_evid.jpg: están anotados con\n"
        "     overlay, recuadros y texto, y ningún modelo fue entrenado con eso.\n"
        "     Validar sobre evidencias anotadas fue el error de la versión anterior."
    )


def run_frame(sess: onnxio.OnnxModel, img: np.ndarray, size: int,
              thresh: int) -> tuple[list[float], str, np.ndarray, np.ndarray]:
    """Devuelve (porcentajes, veredicto, mapa de clases, máscara de válidos)."""
    tensor = PP.preprocess_bgr(img, size)
    logits = sess.run({"input": tensor})
    seg = np.argmax(logits[0], axis=0)

    mask = (ND.border_mask(img, thresh) if thresh > 0
            else np.ones(img.shape[:2], dtype=bool))
    valid = ND.valid_at_size(mask, size)
    pcts, _dom, _n_valid = ND.terrain_percentages(seg, valid, len(IDX.CLASS_NAMES))
    env = IDX.environment(pcts, valid_frac=(valid.mean() if valid.size else 0.0))
    return pcts, env["verdict"], seg, valid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Validación INT8 sobre métricas de misión",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--fp32", default="outputs/cansat_seg_terrain_v2.onnx")
    ap.add_argument("--int8", default="outputs/cansat_seg_terrain_v2_int8.onnx")
    ap.add_argument("--frames", default=None, help="carpeta de frames crudos")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42,
                    help="semilla del muestreo (antes tomaba los primeros N por "
                         "orden alfabético: sesgado y no reproducible)")
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--nodata-thresh", type=int, default=15,
                    help="0 = desactivar la máscara de no-data")
    ap.add_argument("--out", default=None,
                    help="JSON de resultados (default: outputs/metrics/int8_<ts>.json)")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("  Validación INT8 — métricas de misión")
    print("=" * 78)

    fp32 = onnxio.load_required(args.fp32, "FP32")
    int8 = onnxio.load_required(args.int8, "INT8",
                                hint="Generalo con quantize_all.py o quantize_onnx.py.")
    print(f"  FP32 : {fp32.describe()}")
    print(f"  INT8 : {int8.describe()}")

    # ⚠ La comparación válida es MISMO grafo, distinta precisión. Si los ONNX
    # no vienen de la misma exportación, el resultado no significa nada.
    if fp32.input_shape != int8.input_shape:
        print(f"  [WARN] las entradas no coinciden: {fp32.input_shape} vs "
              f"{int8.input_shape}. ¿Son realmente el mismo modelo?")
    # El error del informe viejo: --fp32 y --int8 apuntando al MISMO archivo
    # daba 100 % "APTO". Eso no mide la cuantización: se aborta.
    if Path(args.fp32).resolve() == Path(args.int8).resolve():
        print("\n[ERROR] --fp32 y --int8 son el MISMO archivo. La comparación no "
              "mediría nada (es el error que produjo el '100 % APTO' del informe).")
        return 1
    try:
        import hashlib

        def _sha(p):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(blk)
            return h.hexdigest()

        if _sha(args.fp32) == _sha(args.int8):
            print("\n[ERROR] --fp32 y --int8 tienen el MISMO hash: son el mismo "
                  "modelo. Abortando.")
            return 1
    except OSError:
        pass

    try:
        frames = collect_frames(args)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return 1
    print(f"  Frames crudos a validar: {len(frames)}\n")

    rows = []
    ver_match = alert_match = 0
    pixel_agree: list[float] = []
    mae_pct: list[float] = []

    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            print(f"  [WARN] no se pudo leer {fp.name}, salteando.")
            continue

        pcts_a, verd_a, seg_a, valid = run_frame(fp32, img, args.img_size, args.nodata_thresh)
        pcts_b, verd_b, seg_b, _ = run_frame(int8, img, args.img_size, args.nodata_thresh)

        # El acuerdo se mide SOLO sobre píxeles válidos: incluir el borde negro
        # inflaba artificialmente el número (ambos modelos aciertan en negro).
        agree = float((seg_a == seg_b)[valid].mean()) if valid.any() else 0.0
        pixel_agree.append(agree)
        mae_pct.append(float(np.mean(np.abs(np.array(pcts_a) - np.array(pcts_b)))))

        # Diagnóstico/alerta con los mismos umbrales que el vuelo. Sin modelos de
        # daño acá: el objetivo es aislar el efecto de la cuantización sobre el
        # terreno, que es lo que alimenta el veredicto.
        diag_a, alert_a = IDX.diagnose(pcts_a, 0.0, 0.0, 0.0)
        diag_b, alert_b = IDX.diagnose(pcts_b, 0.0, 0.0, 0.0)

        vm = verd_a == verd_b
        am = alert_a == alert_b
        ver_match += int(vm)
        alert_match += int(am)

        rows.append({
            "frame": fp.name, "pixel_agreement": round(agree, 5),
            "mae_pct_terreno": round(mae_pct[-1], 3),
            "verdict_fp32": verd_a, "verdict_int8": verd_b, "verdict_match": vm,
            "alert_fp32": alert_a, "alert_int8": alert_b, "alert_match": am,
            "diag_fp32": diag_a, "diag_int8": diag_b,
        })
        print(f"  {fp.name:<22} acuerdo px {agree * 100:5.1f}%  "
              f"MAE terreno {mae_pct[-1]:4.1f} pts  "
              f"{verd_a:<20} → {verd_b:<20} {'✓' if vm else '✗'}"
              f"{'' if am else '  ⚠ ALERTA DIVERGE'}")

    if not rows:
        print("\n[ERROR] No se pudo procesar ningún frame. "
              "(La versión anterior dividía por cero acá.)")
        return 1

    n = len(rows)
    m_pixel = float(np.mean(pixel_agree))
    p5_pixel = float(np.percentile(pixel_agree, 5))     # el peor 5 % de frames
    m_ver = ver_match / n
    m_alert = alert_match / n
    m_mae = float(np.mean(mae_pct))

    print("\n" + "=" * 78)
    print(f"  RESULTADO sobre {n} frames crudos (semilla {args.seed})")
    print("-" * 78)
    print(f"  Acuerdo píxel a píxel    : {m_pixel * 100:6.2f}%   (umbral {UMBRAL_ARGMAX * 100:.0f}%)")
    print(f"  Peor 5 % de frames (p5)  : {p5_pixel * 100:6.2f}%   (la media sola esconde outliers)")
    print(f"  Acuerdo de veredicto     : {m_ver * 100:6.2f}%   (umbral {UMBRAL_VEREDICTO * 100:.0f}%)")
    print(f"  Acuerdo de alerta        : {m_alert * 100:6.2f}%   (umbral {UMBRAL_ALERTA * 100:.0f}%)")
    print(f"  MAE de % de terreno      : {m_mae:6.2f} puntos porcentuales")

    apto = (m_pixel >= UMBRAL_ARGMAX and m_ver >= UMBRAL_VEREDICTO
            and m_alert >= UMBRAL_ALERTA)
    print("-" * 78)
    if apto:
        print("  ✅ INT8 APTO PARA VUELO: las métricas de misión no cambian.")
        verdict_txt = "APTO"
    elif m_ver >= UMBRAL_VEREDICTO and m_alert >= UMBRAL_ALERTA:
        print("  ⚠️  INT8 marginal: el argmax por píxel difiere más de lo deseado,")
        print("     pero LOS VEREDICTOS Y LAS ALERTAS coinciden. Para la métrica de")
        print("     misión (que es lo que se evalúa) es usable; para mapas de")
        print("     cobertura detallados, no.")
        verdict_txt = "MARGINAL"
    else:
        divergen = [r for r in rows if not r["verdict_match"] or not r["alert_match"]]
        print(f"  ❌ INT8 NO APTO: {len(divergen)} de {n} frames cambian veredicto o alerta.")
        print("     Opciones: cuantización estática calibrada (quantize_onnx.py,")
        print("     calibrando con TRAIN y validando con VAL, no con Test), QAT, o")
        print("     volar en FP32 si la Pi lo aguanta.")
        verdict_txt = "NO APTO"
    print("=" * 78)

    out = Path(args.out) if args.out else (
        PROJ.METRICS_DIR / f"int8_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "script": "validate_int8_mission.py",
        "fp32": str(args.fp32), "int8": str(args.int8),
        "n_frames": n, "img_size": args.img_size,
        "nodata_thresh": args.nodata_thresh,
        "umbrales": {"pixel": UMBRAL_ARGMAX, "veredicto": UMBRAL_VEREDICTO,
                     "alerta": UMBRAL_ALERTA},
        "acuerdo_pixel": round(m_pixel, 5),
        "acuerdo_pixel_p5": round(p5_pixel, 5),
        "acuerdo_veredicto": round(m_ver, 5),
        "acuerdo_alerta": round(m_alert, 5),
        "mae_pct_terreno": round(m_mae, 4),
        "veredicto_validacion": verdict_txt,
        "frames": rows,
        "nota": ("Validación sobre frames CRUDOS con los índices de "
                 "cansat.indices (los mismos que vuelan). La versión anterior "
                 "validaba sobre evidencias anotadas y con fórmulas distintas; "
                 "su 87.6% no medía la cuantización."),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"  → {out}")
    return 0 if apto else 2


if __name__ == "__main__":
    raise SystemExit(main())
