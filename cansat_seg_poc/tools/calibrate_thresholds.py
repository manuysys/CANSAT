"""
Calibración de umbrales de daño/alerta con frames reales — CanSat LB135.

Responde la pregunta que quedó abierta en la auditoría: *"los umbrales del
consenso (10 % de voto por modelo) nunca se calibraron con frames reales"*.

Cómo lo hace
------------
Corre los MISMOS modelos y la MISMA lógica que el vuelo
(``mission_pipeline``: terreno → máscara de edificios → daño por modelo,
normalizado por píxeles válidos) sobre un conjunto de frames **con etiqueta
conocida**, y reporta:

  · la distribución de ``danado_max_pct`` en frames SIN desastre y CON desastre;
  · el umbral que mejor separa (máximo de F1) y el que fija una tasa de falsos
    positivos objetivo (--max-fpr, default 5 %);
  · los umbrales actuales del repo (``cansat.indices.DAMAGE_CONSENSUS_PCT`` y
    ``DAMAGE_STRONG_PCT``) para comparar.

Fuentes de frames etiquetados
-----------------------------
  · ``--xbd`` (default): usa el manifest de xBD con el **split por desastre**
    (los desastres de val nunca se vieron en train). Etiqueta positiva =
    tile con fracción de píxeles "dañado" (clases 2-4) mayor a
    ``--min-dano-px``.
  · ``--rescuenet``: usa ``dataset/rescuenet_tiles/manifest_val.csv`` (UAV,
    el dominio de vuelo: GSD ≈0.05-0.08 m/px). Etiqueta positiva = tile con
    fracción de "dañado" (clase 2 del remapeo) ≥ ``--min-dano-px``.
  · ``--folder``: carpeta propia con ``--labels`` (CSV ``src,danado`` 0/1).

Uso:
    python tools/calibrate_thresholds.py                 # xBD val, split honesto
    python tools/calibrate_thresholds.py --models all    # incluye two-stage
    python tools/calibrate_thresholds.py --rescuenet \\
        --damage-onnx outputs/cansat_damage3_bal.onnx \\
        --damage2-onnx outputs/cansat_damage_v3_bal.onnx
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cansat import indices as IDX
from cansat import nodata as ND
from cansat import onnxio
from cansat import preprocess as PP
from cansat.xbd import split_por_desastre

MANIFEST = Path("dataset/xbd_masks/manifest.csv")
RESCUENET_MANIFEST = Path("dataset/rescuenet_tiles/manifest_val.csv")
TERRAIN = "outputs/cansat_seg_terrain_v2.onnx"
DAMAGE = "outputs/cansat_damage3_mobilenetv2.onnx"
# Two-stage de vuelo: adaptado a UAV con RescueNet (F2 2026-09-18).
DAMAGE2 = "outputs/cansat_damage_v3_bal.onnx"


def metricas(pcts_dano: np.ndarray, etiquetas: np.ndarray) -> dict:
    """Separa positivos/negativos y calcula F1 por umbral."""
    pos = pcts_dano[etiquetas == 1]
    neg = pcts_dano[etiquetas == 0]
    umbrales = np.unique(np.round(np.concatenate([pcts_dano, [0.0, 100.0]]), 2))
    mejor_f1, mejor_u = -1.0, 0.0
    filas = []
    for u in umbrales:
        tp = int((pos >= u).sum())
        fp = int((neg >= u).sum())
        fn = int((pos < u).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        filas.append((float(u), prec, rec, f1, fp))
        if f1 > mejor_f1:
            mejor_f1, mejor_u = f1, float(u)
    return {"pos": pos, "neg": neg, "mejor_u": mejor_u, "mejor_f1": mejor_f1,
            "filas": filas}


def umbral_para_fpr(metricas_: dict, max_fpr: float) -> tuple[float, float]:
    """Umbral más bajo cuya tasa de falsos positivos no supere ``max_fpr``."""
    neg = metricas_["neg"]
    if neg.size == 0:
        return metricas_["mejor_u"], 0.0
    for u, _p, _r, _f, fp in metricas_["filas"]:
        if fp / neg.size <= max_fpr:
            return u, fp / neg.size
    return 100.0, 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibración de umbrales de daño")
    ap.add_argument("--xbd", action="store_true", default=True,
                    help="usar xBD val con split por desastre (default)")
    ap.add_argument("--rescuenet", action="store_true",
                    help="usar RescueNet val (UAV, dominio de vuelo)")
    ap.add_argument("--folder", default=None, help="carpeta de frames propia")
    ap.add_argument("--labels", default=None, help="CSV src,danado (0/1) para --folder")
    ap.add_argument("--tiles", type=int, default=120, help="tiles de xBD a evaluar")
    ap.add_argument("--min-dano-px", type=float, default=2.0,
                    help="%% de píxeles dañados en el GT para etiquetar el tile como positivo")
    ap.add_argument("--models", choices=("principal", "all"), default="all")
    ap.add_argument("--terrain-onnx", default=TERRAIN)
    ap.add_argument("--damage-onnx", default=DAMAGE)
    ap.add_argument("--damage2-onnx", default=DAMAGE2)
    ap.add_argument("--torch", action="store_true",
                    help="correr en GPU con los checkpoints .pth (equivalente a "
                         "los ONNX, ~1e-6 de diferencia); cv2.dnn/ORT acá son CPU")
    ap.add_argument("--terrain-ckpt", default="outputs/best_terrain_v2.pth")
    ap.add_argument("--damage-ckpt", default="outputs/best_damage3.pth")
    ap.add_argument("--damage2-ckpt", default="outputs/best_damage_v3_bal.pth")
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--nodata-thresh", type=int, default=0,
                    help="0 = sin máscara de nodata (frames de cámara)")
    ap.add_argument("--max-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    print("=" * 74)
    print("  Calibración de umbrales — CanSat LB135")
    print("=" * 74)
    print(f"  Umbrales actuales del repo: voto={IDX.DAMAGE_CONSENSUS_PCT} % · "
          f"fuerte={IDX.DAMAGE_STRONG_PCT} % · agua={IDX.AGUA_EXTENSA_PCT} %")

    if args.torch:
        # Modo GPU: mismos pesos que los ONNX (los .pth son la fuente), pero
        # torch+cu128 en vez de cv2.dnn. Diferencia numérica ~1e-6: los umbrales
        # calibrados son los mismos.
        import torch

        from evaluate import _build_model

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Modo torch: device {dev}")

        class TorchModel:
            def __init__(self, model):
                self.model = model

            def run(self, feeds):
                # Misma convención que onnxio.run(): array (1, C, H, W).
                with torch.no_grad():
                    y = self.model(torch.from_numpy(feeds["input"]).to(dev))
                return y.cpu().numpy()

        def cargar(ckpt: str, n_cls: int):
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            sd = (state.get("model_state_dict", state)
                  if isinstance(state, dict) else state)
            model = _build_model(ckpt, sd, n_cls)
            if model is None:
                raise RuntimeError(f"No pude inferir la arquitectura de {ckpt}")
            model.load_state_dict(sd, strict=False)
            return TorchModel(model.to(dev).eval())

        terr = cargar(args.terrain_ckpt, 5)
        m_d = cargar(args.damage_ckpt, 3)
        m_d2 = cargar(args.damage2_ckpt, 3) if args.models == "all" else None
    else:
        terr = onnxio.load_required(args.terrain_onnx, "terreno")
        m_d = onnxio.load_required(args.damage_onnx, "daño principal")
        m_d2 = (onnxio.load_optional(args.damage2_onnx, "two-stage")
                if args.models == "all" else None)

    frames: list[tuple[str, np.ndarray, int]] = []   # (nombre, bgr, etiqueta)
    if args.folder:
        if not args.labels:
            print("[ERROR] --folder necesita --labels (CSV src,danado)")
            return 1
        with open(args.labels, encoding="utf-8") as fh:
            etiquetas = {r["src"]: int(r["danado"]) for r in csv.DictReader(fh)}
        for p in sorted(Path(args.folder).glob("*")):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.stem in etiquetas:
                img = cv2.imread(str(p))
                if img is not None:
                    frames.append((p.stem, img, etiquetas[p.stem]))
    elif args.rescuenet:
        if not RESCUENET_MANIFEST.is_file():
            print(f"[ERROR] No existe {RESCUENET_MANIFEST}. "
                  f"Corré prepare_rescuenet.py --split val")
            return 1
        with open(RESCUENET_MANIFEST, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        import random
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        for r in rows[:args.tiles]:
            img = cv2.imread(r["image"])
            msk = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                continue
            frac_dano = float((msk == 2).mean()) * 100.0
            etiqueta = 1 if frac_dano >= args.min_dano_px else 0
            frames.append((Path(r["image"]).stem, img, etiqueta))
        print(f"  Fuente: RescueNet val (UAV) — {len(frames)} tiles "
              f"(de {len(rows)}, seed {args.seed})")
    else:
        with open(MANIFEST, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        _tr, val, val_g = split_por_desastre(rows, 0.2, args.seed)
        print(f"  xBD val por desastre: {val_g} ({len(val)} tiles)")
        import random
        rng = random.Random(args.seed)
        rng.shuffle(val)
        for r in val[:args.tiles]:
            img = cv2.imread(r["image"])
            msk = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                continue
            frac_dano = float((msk >= 2).mean()) * 100.0
            etiqueta = 1 if frac_dano >= args.min_dano_px else 0
            frames.append((Path(r["image"]).stem, img, etiqueta))

    if not frames:
        print("[ERROR] No se juntó ningún frame etiquetado.")
        return 1
    n_pos = sum(e for _s, _i, e in frames)
    print(f"  Frames: {len(frames)} ({n_pos} con daño, {len(frames) - n_pos} sin daño)")
    print(f"  Etiqueta positiva = ≥{args.min_dano_px} % de píxeles dañados en el GT\n")

    def mascaras(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Máscaras del vuelo: válidos (nodata) y edificios del terreno."""
        tensor = PP.preprocess_bgr(img, args.img_size)
        seg = np.argmax(terr.run({"input": tensor})[0], axis=0)
        if args.nodata_thresh > 0:
            valid = ND.valid_at_size(ND.border_mask(img, 15), args.img_size)
        else:
            valid = np.ones_like(seg, dtype=bool)
        return valid, (seg == 1) & valid

    def dano_por_frame(img: np.ndarray, modelo: onnxio.OnnxModel,
                       valid: np.ndarray, bmask: np.ndarray) -> float:
        tensor = PP.preprocess_bgr(img, args.img_size)
        n_valid = max(1, int(valid.sum()))
        pred = np.argmax(modelo.run({"input": tensor})[0], axis=0)
        if modelo is m_d2:
            # El two-stage de la misión se enmascara con los edificios del
            # terreno (si no, predice daño en cualquier lado).
            return float(((pred == 2) & bmask).sum()) / n_valid * 100.0
        return float(((pred == 2) & valid).sum()) / n_valid * 100.0

    nombres = ["principal"] + (["two-stage"] if m_d2 else [])
    for nombre, modelo in zip(nombres, [m_d, m_d2], strict=False):
        if modelo is None:
            continue
        valores = []
        for _s, img, _e in frames:
            valid, bmask = mascaras(img)
            valores.append(dano_por_frame(img, modelo, valid, bmask))
        valores = np.array(valores)
        etiquetas = np.array([e for _s, _i, e in frames])
        met = metricas(valores, etiquetas)
        u_fpr, fpr = umbral_para_fpr(met, args.max_fpr)
        fila_f1 = next((f for f in met["filas"] if f[0] == met["mejor_u"]), None)
        print("-" * 74)
        print(f"  Modelo: {nombre}")
        print(f"    daño en frames SIN desastre: mediana {np.median(met['neg']):.2f} % "
              f"· p95 {np.percentile(met['neg'], 95):.2f} % · máx {met['neg'].max():.2f} %")
        print(f"    daño en frames CON desastre: mediana {np.median(met['pos']):.2f} % "
              f"· p95 {np.percentile(met['pos'], 95):.2f} %")
        if fila_f1:
            _u, prec, rec, f1, fp = fila_f1
            print(f"    mejor F1: {f1:.3f} con umbral {met['mejor_u']:.2f} % "
                  f"(precisión {prec:.3f} · recall {rec:.3f} · "
                  f"FPR {fp / max(1, met['neg'].size):.1%})")
        print(f"    umbral para FPR ≤ {args.max_fpr:.0%}: {u_fpr:.2f} % "
              f"(FPR real {fpr:.1%})")
        print(f"    → sugerido para --damage-threshold: {u_fpr:.1f}")

    print("-" * 74)
    print("  Cómo usarlo: mission_pipeline.py --damage-threshold <umbral>.")
    print("  El JSONL de cada vuelo registra danado_* por frame: recalibrá con")
    print("  frames reales del predio cuando los tengas (--folder/--labels).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
