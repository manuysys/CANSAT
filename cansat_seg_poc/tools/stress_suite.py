"""
Stress suite limpio-vs-estresado — aptitud de vuelo medida, no declarada.

Aplica las corrupciones sintéticas de ``cansat/corrupt.py`` a sets de test ya
establecidos y reporta F1/IoU limpio vs estresado por corrupción. Es el número
honesto que pide el DPD: el mismo modelo, la misma GT, sólo cambia el clima.

Tareas:
  · ``terreno``  ``cansat_seg_terrain_v2.onnx`` × LoveDA Val (5 clases)
  · ``dano``     ``cansat_damage3_mobilenetv2.onnx`` (xBD) × Joplin/Nepal
                 held-out (eventos que NUNCA entrenaron ese checkpoint)
  · ``dano2``    ``cansat_damage_v3_bal.onnx`` (two-stage de vuelo) ×
                 RescueNet val (dominio UAV, GSD de vuelo)
  · ``flood``    ``cansat_flood_specialist_224.onnx`` × FloodNet val (80 imgs)
  · ``fuego``    ``cansat_fire_smoke.onnx`` × fire_smoke valid+test (60 imgs)

Uso:
    python tools/stress_suite.py
    python tools/stress_suite.py --tasks terreno --max-images 200
    python tools/stress_suite.py --tasks dano2 --corruptions niebla,motion_blur
    python tools/stress_suite.py --tasks flood,fuego

Salida: ``outputs/stress_suite.json`` con hashes de modelo, seed, n de imagen,
métricas limpias y por corrupción + deltas, parámetros usados y la nota de
honestidad (corrupción sintética, no validación de vuelo real).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402

from cansat import corrupt as COR                             # noqa: E402
from cansat import indices as IDX                             # noqa: E402
from cansat import metrics as MET                             # noqa: E402
from cansat import onnxio                                     # noqa: E402
from cansat import preprocess as PP                           # noqa: E402
from evaluate import collect, file_hash, remap_raw            # noqa: E402

NOMBRES_DANO: tuple[str, ...] = ("other", "intacto", "danado")
NOMBRES_FLOOD: tuple[str, ...] = ("other", "inundacion", "agua")
NOMBRES_FUEGO: tuple[str, ...] = ("other", "fuego", "humo")
TAREAS_ORDEN: tuple[str, ...] = ("terreno", "dano", "dano2", "flood", "fuego")

#: Métrica cabecera por tarea (clave del resumen y etiqueta para imprimir).
OBJETIVO = {
    "terreno": ("miou", "mIoU"),
    "dano": ("iou_bin_dano", "IoU daño bin"),
    "dano2": ("iou_bin_dano", "IoU daño bin"),
    "flood": ("iou_clase_objetivo", "IoU inundación"),
    "fuego": ("iou_clase_objetivo", "IoU fuego"),
}

NOTA = (
    "Corrupción SINTÉTICA con severidad fija (parámetros en cansat/corrupt.py). "
    "No reemplaza validación de vuelo real: mide robustez relativa del mismo "
    "modelo contra sí mismo. La niebla se verifica con haze_metrics."
)


# ══════════════════════════════════════════════════════════════════════════ #
#  Datos
# ══════════════════════════════════════════════════════════════════════════ #
def pares_terreno(root: Path, splits: list[str], max_images: int, seed: int):
    """Pares (imagen, máscara, remapeada) de LoveDA Val, muestreo reproducible."""
    pares, info = collect(root, splits, max_images, seed)
    return pares, info


def pares_manifest(manifests: list[str], max_images: int, seed: int):
    """Pares (imagen, máscara) de manifests formato xBD, muestreo reproducible."""
    filas: list[dict] = []
    for m in manifests:
        with open(m, encoding="utf-8") as f:
            filas += list(csv.DictReader(f))
    filas = [r for r in filas
             if Path(r["image"]).is_file() and Path(r["mask"]).is_file()]
    filas.sort(key=lambda r: r["name"])
    if max_images and len(filas) > max_images:
        idx = sorted(np.random.RandomState(seed)
                     .permutation(len(filas))[:max_images].tolist())
        filas = [filas[i] for i in idx]
    return [(Path(r["image"]), Path(r["mask"])) for r in filas]


def xbd_encode(mask: np.ndarray) -> np.ndarray:
    """xBD 4 clases → 3 (0 other, 1 intacto, 2 dañado), igual que XBDv3."""
    return np.where(mask >= 2, 2, mask).astype(np.uint8)


def pares_floodnet(root: Path, max_images: int, seed: int):
    """Pares (imagen, máscara) del val remapeado de FloodNet (clases 0/1/2)."""
    base = root / "floodnet_remapped" / "val"
    pares = [(p, base / "masks" / p.name) for p in sorted((base / "images").glob("*.png"))
             if (base / "masks" / p.name).is_file()]
    if max_images and len(pares) > max_images:
        idx = sorted(np.random.RandomState(seed)
                     .permutation(len(pares))[:max_images].tolist())
        pares = [pares[i] for i in idx]
    return pares


def nombres_clases(tarea: str, n_cls: int) -> list[str]:
    if tarea == "terreno" and n_cls == IDX.NUM_CLASSES:
        return list(IDX.CLASS_NAMES[:n_cls])
    if tarea in ("dano", "dano2") and n_cls == len(NOMBRES_DANO):
        return list(NOMBRES_DANO[:n_cls])
    if tarea == "flood" and n_cls == len(NOMBRES_FLOOD):
        return list(NOMBRES_FLOOD[:n_cls])
    if tarea == "fuego" and n_cls == len(NOMBRES_FUEGO):
        return list(NOMBRES_FUEGO[:n_cls])
    return [f"clase_{i}" for i in range(n_cls)]


# ══════════════════════════════════════════════════════════════════════════ #
#  Evaluación
# ══════════════════════════════════════════════════════════════════════════ #
def _promediar(metas: list[dict]) -> dict:
    """Promedio de los metadatos por imagen (niebla y escala varían por frame)."""
    if not metas:
        return {}
    out: dict = {}
    for k in metas[0]:
        vals = [m[k] for m in metas if k in m]
        if not vals:
            continue
        if isinstance(vals[0], int | float):
            out[k] = round(float(np.mean(vals)), 4)
        elif isinstance(vals[0], list):
            out[k] = [round(float(np.mean([v[j] for v in vals])), 4)
                      for j in range(len(vals[0]))]
        else:
            out[k] = vals[0]
    return out


def evaluar(sess, pares, mask_mode: str, n_cls: int, img_size: int,
            corrupcion: str | None, rng, tarea: str,
            log_cada: int = 50, etiqueta: str = ""):
    """
    Corre el modelo sobre ``pares`` con (o sin) corrupción y devuelve
    ``(resumen, conf, n)``. Reusa la métrica única de ``cansat.metrics``.
    """
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)
    nitideces: list[float] = []
    metas: list[dict] = []
    t0 = time.time()
    n = 0
    for i, par in enumerate(pares):
        if mask_mode == "terreno":
            ip, mp = par[0], par[1]
            remapeada = par[2]
        else:
            ip, mp = par
            remapeada = True

        bgr = cv2.imread(str(ip))
        if mask_mode == "terreno":
            mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        else:
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask_mode == "terreno" and not remapeada:
            mask = remap_raw(mask)
        elif mask_mode == "dano":
            mask = xbd_encode(mask)

        if corrupcion is not None:
            bgr, mask, meta = COR.aplicar(corrupcion, bgr, mask, rng)
            metas.append(meta)
        nitideces.append(COR.nitidez(bgr))

        x = PP.preprocess_bgr(bgr, img_size)
        pred = np.argmax(sess.run({"input": x})[0], axis=0)
        mask_s = cv2.resize(mask.astype(np.uint8), (img_size, img_size),
                            interpolation=cv2.INTER_NEAREST)
        MET.accumulate(conf, mask_s, pred, n_cls, IDX.IGNORE_INDEX)
        n += 1
        if log_cada and (i + 1) % log_cada == 0:
            el = time.time() - t0
            print(f"      {etiqueta}{i + 1}/{len(pares)}  "
                  f"({(i + 1) / max(el, 1e-9):.1f} img/s)", flush=True)

    resumen = resumen_metricas(conf, n_cls, nitideces, tarea)
    if metas:
        resumen["parametros"] = _promediar(metas)
    return resumen, conf, n


def _iou(conf: np.ndarray, i: int) -> float:
    inter = int(conf[i, i])
    union = int(conf[i, :].sum() + conf[:, i].sum() - inter)
    return inter / union if union else 0.0


def resumen_metricas(conf: np.ndarray, n_cls: int,
                     nitideces: list[float], tarea: str = "terreno") -> dict:
    """
    Métricas del contrato (mIoU/F1 por clase) + la cabecera de cada tarea:
      · dano/dano2 → IoU binaria de daño y two-stage (sobre edificios)
      · flood      → IoU de inundación (clase 1) y de agua normal (clase 2)
      · fuego      → IoU de fuego (clase 1) y humo (clase 2)
    """
    out = MET.from_confusion(conf).as_dict(nombres_clases(tarea, n_cls))
    out["nitidez_media"] = round(float(np.mean(nitideces)), 2) if nitideces else None

    if tarea in ("dano", "dano2") and n_cls == 3:
        inter_bin = int(conf[2, 2])
        union_bin = int(conf[2, :].sum() + conf[:, 2].sum() - inter_bin)
        # Two-stage: dañado predicho DENTRO de edificios vs GT dañado —
        # la métrica con la que se entrena/evalúa el two-stage (xBDv3).
        inter_d = int(conf[2, 2])
        union_d = int(conf[1, 2] + conf[2, 2] + conf[2, 0] + conf[2, 1])
        prec = inter_bin / max(int(conf[:, 2].sum()), 1)
        rec = inter_bin / max(int(conf[2, :].sum()), 1)
        out["iou_bin_dano"] = round(inter_bin / max(union_bin, 1), 6)
        out["f1_bin_dano"] = round(2 * prec * rec / max(prec + rec, 1e-12), 6)
        out["iou_dano_two_stage"] = round(inter_d / max(union_d, 1), 6)
    elif tarea == "flood" and n_cls >= 3:
        out["iou_clase_objetivo"] = round(_iou(conf, 1), 6)
        out["iou_agua"] = round(_iou(conf, 2), 6)
    elif tarea == "fuego" and n_cls >= 3:
        out["iou_clase_objetivo"] = round(_iou(conf, 1), 6)
        out["iou_fuego"] = round(_iou(conf, 1), 6)
        out["iou_humo"] = round(_iou(conf, 2), 6)
    return out


def _delta(estresado: dict, limpio: dict) -> dict:
    """Deltas de las métricas cabecera (negativo = empeoró)."""
    out = {}
    for k in ("miou", "pixel_acc", "iou_bin_dano", "f1_bin_dano",
              "iou_dano_two_stage", "iou_clase_objetivo", "iou_fuego", "iou_humo"):
        if k in estresado and k in limpio:
            out[f"delta_{k}"] = round(estresado[k] - limpio[k], 6)
    if limpio.get("nitidez_media"):
        out["delta_nitidez"] = round(
            (estresado.get("nitidez_media") or 0.0) - limpio["nitidez_media"], 2)
    return out


# ══════════════════════════════════════════════════════════════════════════ #
#  Main
# ══════════════════════════════════════════════════════════════════════════ #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stress suite limpio-vs-estresado (aptitud de vuelo)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tasks", default="terreno,dano,dano2",
                    help="tareas separadas por coma (terreno, dano, dano2, flood, fuego)")
    ap.add_argument("--dataset-root", default="dataset")
    ap.add_argument("--onnx-terreno", default="outputs/cansat_seg_terrain_v2.onnx")
    ap.add_argument("--onnx-dano", default="outputs/cansat_damage3_mobilenetv2.onnx")
    ap.add_argument("--onnx-dano2", default="outputs/cansat_damage_v3_bal.onnx")
    ap.add_argument("--onnx-flood", default="outputs/cansat_flood_specialist_224.onnx")
    ap.add_argument("--onnx-fuego", default="outputs/cansat_fire_smoke.onnx")
    ap.add_argument("--manifests-dano", nargs="+",
                    default=["dataset/xbd_masks/manifest_joplin-tornado.csv",
                             "dataset/xbd_masks/manifest_nepal-flooding.csv"])
    ap.add_argument("--manifest-dano2",
                    default="dataset/rescuenet_tiles/manifest_val.csv")
    ap.add_argument("--manifests-fuego", nargs="+",
                    default=["dataset/fire_smoke/manifest_valid.csv",
                             "dataset/fire_smoke/manifest_test.csv"],
                    help="valid participó de la selección del checkpoint; se declara")
    ap.add_argument("--splits", default="Rural,Urban")
    ap.add_argument("--max-images", type=int, default=120,
                    help="máximo de imágenes por corrida (0 = todas)")
    ap.add_argument("--img-size", type=int, default=PP.DEFAULT_IMG_SIZE)
    ap.add_argument("--corruptions", default="todas",
                    help="todas o lista separada por coma")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/stress_suite.json")
    ap.add_argument("--no-json", action="store_true")
    args = ap.parse_args(argv)

    tareas = [t.strip() for t in args.tasks.split(",") if t.strip()]
    validas = {"terreno", "dano", "dano2", "flood", "fuego"}
    if unknown := [t for t in tareas if t not in validas]:
        print(f"[ERROR] tarea(s) desconocida(s): {unknown}. Soportadas: {sorted(validas)}")
        return 1
    if args.corruptions.strip() == "todas":
        corrupciones = list(COR.CORRUPCIONES)
    else:
        corrupciones = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    if unknown := [c for c in corrupciones if c not in COR.CORRUPCIONES]:
        print(f"[ERROR] corrupción(es) desconocida(s): {unknown}. "
              f"Soportadas: {list(COR.CORRUPCIONES)}")
        return 1

    root = Path(args.dataset_root)
    resultados: dict[str, dict] = {}
    n_por_tarea: dict[str, int] = {}

    print("=" * 74)
    print("  STRESS SUITE — limpio vs estresado (misma GT, mismo modelo)")
    print("=" * 74)
    print(f"  Corrupciones: {', '.join(corrupciones)}")
    print(f"  Muestra: {args.max_images or 'completa'} imgs/tarea · seed {args.seed}\n")

    for tarea in tareas:
        if tarea == "terreno":
            onnx = args.onnx_terreno
            splits = [s.strip() for s in args.splits.split(",") if s.strip()]
            pares, info = pares_terreno(root, splits, args.max_images, args.seed)
            mask_mode = "terreno"
            esperadas = IDX.NUM_CLASSES
            fuente = (f"LoveDA Val {splits} · "
                      f"{info.get('n_evaluadas', len(pares))} imgs "
                      f"({info.get('mascaras')})")
        elif tarea == "dano":
            onnx = args.onnx_dano
            pares = pares_manifest(args.manifests_dano, args.max_images, args.seed)
            mask_mode = "dano"
            esperadas = 3
            fuente = f"{len(args.manifests_dano)} manifests xBD held-out"
        elif tarea == "dano2":
            onnx = args.onnx_dano2
            pares = pares_manifest([args.manifest_dano2], args.max_images, args.seed)
            mask_mode = "dano"
            esperadas = 3
            fuente = f"{Path(args.manifest_dano2).name} (RescueNet val, UAV)"
        elif tarea == "flood":
            onnx = args.onnx_flood
            pares = pares_floodnet(root, args.max_images, args.seed)
            mask_mode = "flood"
            esperadas = 3
            fuente = "FloodNet val (remapeo oficial: flood=1 / agua=2)"
        else:  # fuego
            onnx = args.onnx_fuego
            pares = pares_manifest(args.manifests_fuego, args.max_images, args.seed)
            mask_mode = "fuego"
            esperadas = 3
            fuente = ("fire_smoke valid+test; valid participó de la selección "
                      "del checkpoint (se declara)")

        if not pares:
            print(f"  [WARN] {tarea}: sin pares imagen/máscara — se saltea.\n")
            continue

        sess = onnxio.load_required(onnx, f"stress {tarea}")
        img_size = sess.size_px or args.img_size
        n_cls = sess.n_classes or esperadas
        if n_cls != esperadas:
            print(f"  [WARN] {tarea}: el ONNX declara {n_cls} clases; "
                  f"la tarea espera {esperadas}.")

        print(f"  ── {tarea} ─────────────────────────────────────────────")
        print(f"     modelo : {Path(onnx).name}  sha256:{file_hash(onnx)[:16]}")
        print(f"     fuente : {fuente}")
        print(f"     entrada: {img_size}px · {n_cls} clases · {len(pares)} imgs")

        limpio, _conf, n = evaluar(sess, pares, mask_mode, n_cls, img_size,
                                   None, None, tarea, etiqueta=f"{tarea} limpio ")
        n_por_tarea[tarea] = n
        obj_key, obj_txt = OBJETIVO[tarea]
        obj_val = limpio.get(obj_key)
        print(f"     limpio : mIoU {limpio['miou'] * 100:5.1f}% · "
              f"pixel-acc {limpio['pixel_acc'] * 100:5.1f}% · "
              f"nitidez {limpio['nitidez_media']}"
              + (f" · {obj_txt} {obj_val:.3f}" if obj_val is not None else ""))

        estresado: dict[str, dict] = {}
        for k, corr in enumerate(corrupciones):
            rng = np.random.default_rng([args.seed, TAREAS_ORDEN.index(tarea), k])
            res, _c, _n = evaluar(sess, pares, mask_mode, n_cls, img_size,
                                  corr, rng, tarea, etiqueta=f"{tarea} {corr} ")
            res.update(_delta(res, limpio))
            estresado[corr] = res
            extra = (f" · {obj_txt} {res[obj_key]:.3f} "
                     f"({res[f'delta_{obj_key}']:+.3f})"
                     if res.get(obj_key) is not None else "")
            print(f"     {corr:<15} mIoU {res['miou'] * 100:5.1f}% "
                  f"({res['delta_miou'] * 100:+5.1f}) · "
                  f"nitidez {res['nitidez_media']:9.1f}{extra}")
        print()

        resultados[tarea] = {
            "modelo": str(onnx),
            "modelo_sha256_16": file_hash(onnx),
            "num_clases": n_cls,
            "nombres_clases": nombres_clases(tarea, n_cls),
            "img_size": img_size,
            "fuente": fuente,
            "limpio": limpio,
            "estresado": estresado,
        }

    if not resultados:
        print("[ERROR] ninguna tarea corrió.")
        return 1

    payload = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "script": "tools/stress_suite.py",
        "seed": args.seed,
        "max_images": args.max_images,
        "img_size": args.img_size,
        "corruptions": corrupciones,
        "n_images": n_por_tarea,
        "resultados": resultados,
        "nota": NOTA,
    }
    print("=" * 74)
    print(f"  Resumen: {n_por_tarea} imágenes evaluadas por tarea")

    if not args.no_json:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
