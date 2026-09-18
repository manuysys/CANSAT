"""
Evaluación cuantitativa UNIFICADA de los modelos de segmentación.

Reemplaza a tres scripts que hacían lo mismo de tres maneras distintas:

  · ``evaluate_val.py``    ONNX sobre LoveDA Val. Por default evaluaba **sólo
                           100 imágenes** de ~1669 e imprimía al final
                           ``(mIoU de entrenamiento reportado: 52.44%)`` — una
                           constante hardcodeada.
  · ``eval_onnx_gap.py``   docstring *"Compara mIoU en Val: checkpoint FP32 vs
                           inferencia cuantizada"*; el código **sólo evaluaba el
                           FP32**. La comparación no existía. Matriz de confusión
                           con doble loop de Python.
  · ``eval_int8_cpu.py``   sin ``main()``, sin argparse, todo a nivel módulo,
                           ``1669`` hardcodeado en la estimación, y un ``print``
                           por cada uno de los 50 frames.

En el repo convivían **cuatro mIoU distintos** para el mismo modelo
(46.36% en el informe, 52.44% en un print, 0.5317 en otro print y "~0.5317" en
un comentario). Ninguno se calculaba en runtime.

Éste escribe todo a ``outputs/metrics/val_<timestamp>.json`` con el hash del
modelo, la semilla, el tamaño real de la muestra y las métricas por clase.
``generate_report.py`` lo lee. **Todo número citado en el DPD debe salir de
uno de esos JSON.**

Uso:
    python evaluate.py                                  # Val completo, ONNX de vuelo
    python evaluate.py --onnx outputs/cansat_seg_terrain_v2.onnx --max-images 200
    python evaluate.py --checkpoint outputs/best_terrain_cbam.pth   # PyTorch
    python evaluate.py --compare outputs/..._int8.onnx  # FP32 vs INT8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
from cansat import indices as IDX
from cansat import metrics as MET
from cansat import onnxio
from cansat import paths as PROJ
from cansat import preprocess as PP

# LoveDA raw (8 clases) → 5 clases. Debe coincidir con remap_loveda.py.
RAW_TO_5 = {0: 4, 1: 1, 2: 4, 3: 2, 4: 3, 5: 0, 6: 0}


# ══════════════════════════════════════════════════════════════════════ #
#  Dataset
# ══════════════════════════════════════════════════════════════════════ #
def find_pairs(env: str, root: Path) -> tuple[Path | None, Path | None, bool]:
    """Devuelve ``(img_dir, mask_dir, es_remapeada)`` para un split de Val."""
    img_dir = root / "loveda_raw" / "Val" / env / "images_png"
    for cand in (root / "loveda_remapped" / "Val" / env / "masks_png",
                 root / "loveda_remapped" / env / "masks_png",
                 root / "loveda_remapped" / "val" / env / "masks_png"):
        if cand.is_dir():
            return img_dir, cand, True
    raw = root / "loveda_raw" / "Val" / env / "masks_png"
    if raw.is_dir():
        return img_dir, raw, False
    return None, None, False


def collect(root: Path, splits: list[str], max_images: int,
            seed: int) -> tuple[list[tuple[Path, Path, bool]], dict]:
    """Arma la lista de pares (imagen, máscara) con muestreo reproducible."""
    pairs: list[tuple[Path, Path, bool]] = []
    info: dict = {"splits": {}, "mascaras": None}
    for env in splits:
        img_dir, mask_dir, remapped = find_pairs(env, root)
        if img_dir is None or not img_dir.is_dir() or mask_dir is None:
            print(f"  [WARN] Val/{env} no encontrado en {root}, salteando.")
            info["splits"][env] = {"encontrado": False}
            continue
        imgs = sorted(img_dir.glob("*.png"))
        info["splits"][env] = {"encontrado": True, "disponibles": len(imgs),
                               "remapeadas": remapped}
        info["mascaras"] = "remapeadas" if remapped else "raw (remapeo al vuelo)"
        pairs += [(p, mask_dir / p.name, remapped) for p in imgs
                  if (mask_dir / p.name).is_file()]

    if not pairs:
        return [], info

    # Muestreo reproducible: permutación con semilla sobre una lista ORDENADA.
    # (La versión anterior hacía rng.shuffle() sobre una lista de Path, que no
    #  es estable entre versiones de NumPy.)
    if max_images and len(pairs) > max_images:
        order = np.random.RandomState(seed).permutation(len(pairs))
        idx = sorted(order[:max_images].tolist())
        pairs = [pairs[i] for i in idx]
    info["n_evaluadas"] = len(pairs)
    return pairs, info


def remap_raw(mask: np.ndarray) -> np.ndarray:
    """LoveDA raw (8 clases) → 5 clases. Los no-data (255) se preservan."""
    out = np.full(mask.shape, IDX.IGNORE_INDEX, dtype=np.uint8)
    for src, dst in RAW_TO_5.items():
        out[mask == src] = dst
    return out


# ══════════════════════════════════════════════════════════════════════ #
#  Evaluación
# ══════════════════════════════════════════════════════════════════════ #
def eval_onnx(model_path: str, pairs, img_size: int, batch_log: int = 100,
              n_cls: int = IDX.NUM_CLASSES):
    sess = onnxio.load_required(model_path, "evaluación")
    size = sess.size_px or img_size
    if size != img_size:
        print(f"  [i] El modelo fija la entrada a {size}px; se ignora --img-size.")
        img_size = size
    if sess.n_classes is not None and sess.n_classes != n_cls:
        print(f"  [i] El ONNX declara {sess.n_classes} clases; se usa ese número "
              f"en la matriz (--num-classes decía {n_cls}).")
        n_cls = sess.n_classes

    conf = np.zeros((n_cls, n_cls), dtype=np.int64)
    preds: list[np.ndarray] = []
    t0 = time.time()
    for i, (ip, mp, remapped) in enumerate(pairs):
        img = cv2.imread(str(ip))
        mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if not remapped:
            mask = remap_raw(mask)

        x = PP.preprocess_bgr(img, img_size)
        # onnxio.run() ya devuelve (1,C,H,W): un solo [0] para el eje batch.
        # Antes había un [0][0] extra y quedaba (H,W) de la clase 0.
        pred = np.argmax(sess.run({"input": x})[0], axis=0)
        mask_s = cv2.resize(mask.astype(np.uint8), (img_size, img_size),
                            interpolation=cv2.INTER_NEAREST)
        MET.accumulate(conf, mask_s, pred, n_cls, IDX.IGNORE_INDEX)
        preds.append(pred)

        if batch_log and (i + 1) % batch_log == 0:
            el = time.time() - t0
            print(f"    {i + 1}/{len(pairs)}  ({(i + 1) / el:.1f} img/s)", flush=True)

    return conf, preds, time.time() - t0, img_size


def eval_torch(ckpt_path: str, pairs, img_size: int, batch_log: int = 100,
               arch: str | None = None, n_cls: int = IDX.NUM_CLASSES):
    """Evalúa un checkpoint PyTorch con la misma métrica que el ONNX."""
    import torch
    from cansat.seed import device as pick_device

    dev = pick_device("cuda")
    print(f"  [i] device: {dev} (antes estaba hardcodeado a cuda y no corría sin NVIDIA)")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = state.get("model_state_dict", state) if isinstance(state, dict) else state

    model = _build_model(ckpt_path, sd, n_cls, arch)
    if model is None:
        raise RuntimeError(
            f"No pude determinar la arquitectura para {ckpt_path}.\n"
            f"  → Pasá --arch (deeplabv3plus | cbam) explícitamente."
        )
    model.load_state_dict(sd, strict=False)
    model = model.to(dev).eval()

    conf = np.zeros((n_cls, n_cls), dtype=np.int64)
    preds: list[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for i, (ip, mp, remapped) in enumerate(pairs):
            img = cv2.imread(str(ip))
            mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if img is None or mask is None:
                continue
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            if not remapped:
                mask = remap_raw(mask)
            x = torch.from_numpy(PP.preprocess_bgr(img, img_size)).to(dev)
            pred = model(x).argmax(1).cpu().numpy()[0]
            mask_s = cv2.resize(mask.astype(np.uint8), (img_size, img_size),
                                interpolation=cv2.INTER_NEAREST)
            MET.accumulate(conf, mask_s, pred, n_cls, IDX.IGNORE_INDEX)
            preds.append(pred)
            if batch_log and (i + 1) % batch_log == 0:
                print(f"    {i + 1}/{len(pairs)}", flush=True)
    return conf, preds, time.time() - t0, img_size


def _build_model(ckpt_path: str, sd, n_cls: int, arch: str | None = None):
    """Arquitectura del checkpoint: por ``--arch`` o inferida de los tensores."""
    keys = set(sd.keys())
    if arch in (None, "auto"):
        if any(k.startswith("segformer.") or "decode_head" in k for k in keys):
            # SegFormer requiere HuggingFace y su propio loader (train_segformer_b5).
            return None
        if any(k.startswith(("low_classifier.", "high_classifier.")) for k in keys):
            arch = "lraspp"
        else:
            arch = "cbam" if any("aspp" in k and "cbam" in k.lower() for k in keys) \
                else ("deeplabv3plus" if any(k.startswith("aspp.") for k in keys) else None)
    try:
        if arch == "cbam":
            from train_cbam import DeepLabV3PlusCBAM
            return DeepLabV3PlusCBAM(n_cls)
        if arch == "lraspp":
            from train_terrain_tiny import LRASPPMobileNetV3Small
            return LRASPPMobileNetV3Small(n_cls)
        if arch == "deeplabv3plus":
            from train import DeepLabV3PlusMobileNetV2
            return DeepLabV3PlusMobileNetV2(n_cls)
    except ImportError:
        return None
    return None


def file_hash(path: str | Path, limit_mb: float = 64.0) -> str:
    """
    Hash corto del archivo (o del par modelo + pesos externos).

    ⚠ Si el ONNX usa pesos externos (``<modelo>.onnx.data``), el hash del
      ``.onnx`` solo NO identifica al modelo: el grafo de 0.3 MB es idéntico
      entre dos exports distintos. Antes el hash cubría solo el ``.onnx``, así
      que dos versiones con el mismo grafo pero pesos distintos daban la misma
      huella en el JSON de métricas.
    """
    p = Path(path)
    if not p.is_file():
        return "?"
    h = hashlib.sha256()
    archivos = [p]
    sidecar = p.with_name(p.name + ".data")
    if sidecar.is_file():
        h.update(f"data:{sidecar.stat().st_size}".encode())
        archivos.append(sidecar)
    for f in archivos:
        with f.open("rb") as fh:
            read = 0
            while read < limit_mb * 1e6:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
                read += len(chunk)
    return h.hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════ #
#  Main
# ══════════════════════════════════════════════════════════════════════ #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluación cuantitativa unificada",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--onnx", default="outputs/cansat_seg_deeplabv3plus_mobilenetv2.onnx")
    ap.add_argument("--checkpoint", default=None,
                    help="evaluar un .pth con PyTorch en vez del ONNX")
    ap.add_argument("--compare", default=None,
                    help="segundo ONNX a comparar (típicamente la versión INT8)")
    ap.add_argument("--dataset-root", default="dataset")
    ap.add_argument("--splits", default="Rural,Urban")
    ap.add_argument("--max-images", type=int, default=0,
                    help="0 = TODAS las disponibles. El default anterior era 100 "
                         "sobre ~1669, y el informe citaba ese número sin decirlo.")
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--arch", default="auto",
                    choices=("auto", "deeplabv3plus", "cbam", "lraspp"),
                    help="arquitectura del checkpoint (el error la pedía pero el "
                         "flag no existía)")
    ap.add_argument("--num-classes", type=int, default=IDX.NUM_CLASSES,
                    help="clases del modelo evaluado (5 = terreno; 3 = daño)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="archivo .json o carpeta (se agrega val_<timestamp>.json)")
    ap.add_argument("--no-json", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.dataset_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print("=" * 70)
    print("  EVALUACIÓN — LoveDA Val")
    print("=" * 70)
    pairs, info = collect(root, splits, args.max_images, args.seed)
    if not pairs:
        print(f"\n[ERROR] No se encontró ninguna pareja imagen/máscara en {root}.")
        print("        Estructura esperada:")
        print(f"          {root}/loveda_remapped/Val/{{Rural,Urban}}/{{images,masks}}_png/")
        print("        Descargá y remapeá el dataset:")
        print("          python download_loveda.py && python remap_loveda.py")
        return 1
    print(f"  Máscaras: {info.get('mascaras')}")
    print(f"  Muestra : {len(pairs)} imágenes "
          f"(seed {args.seed}) de {sum(s.get('disponibles', 0) for s in info['splits'].values())} disponibles")

    target = args.checkpoint or args.onnx
    print(f"\n  Modelo  : {target}  (sha256:{file_hash(target)[:16]})")
    if args.checkpoint:
        conf, preds, elapsed, size = eval_torch(
            target, pairs, args.img_size, arch=args.arch, n_cls=args.num_classes)
    else:
        conf, preds, elapsed, size = eval_onnx(
            target, pairs, args.img_size, n_cls=args.num_classes)
    nombres = list(IDX.CLASS_NAMES[:args.num_classes])
    m = MET.from_confusion(conf)
    MET.print_table(m, nombres,
                    f"Resultado ({len(preds)} imágenes, {elapsed:.1f} s, "
                    f"{len(preds) / max(elapsed, 1e-9):.1f} img/s)")

    payload: dict = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "script": "evaluate.py",
        "tipo": "checkpoint_pytorch" if args.checkpoint else "onnx",
        "modelo": str(target),
        "modelo_sha256_16": file_hash(target),
        "arquitectura": args.arch,
        "num_clases": args.num_classes,
        "dataset_root": str(root),
        "splits": splits,
        "n_disponibles": sum(s.get("disponibles", 0) for s in info["splits"].values()),
        "n_images": len(preds),
        "img_size": size,
        "seed": args.seed,
        "segundos": round(elapsed, 2),
        "metrics": m.as_dict(nombres),
        "mascaras": info.get("mascaras"),
        "nota": ("Muestra completa." if not args.max_images else
                 f"Muestra acotada a {args.max_images} imágenes: NO es el Val completo."),
    }

    # ── Comparación FP32 vs INT8 ────────────────────────────────────────
    if args.compare:
        print(f"\n  Comparando contra: {args.compare}")
        if file_hash(args.compare) == payload["modelo_sha256_16"] and not args.checkpoint:
            print("  [ERROR] El modelo de comparación es EL MISMO archivo que el "
                  "principal (mismo hash).\n"
                  "          La comparación no mediría nada — es el error que produjo "
                  "el '100% APTO' del informe viejo.")
            return 1
        conf_b, preds_b, _el_b, _ = eval_onnx(args.compare, pairs, size,
                                              n_cls=args.num_classes)
        m_b = MET.from_confusion(conf_b)
        MET.print_table(m_b, nombres, "Modelo de comparación")
        agree = (float(np.mean([MET.agreement(a, b)
                                for a, b in zip(preds, preds_b, strict=False)]))
                 if preds and preds_b else 0.0)
        print(f"\n  Acuerdo píxel a píxel: {agree * 100:.2f}%")
        print(f"  Δ mIoU: {(m_b.miou - m.miou) * 100:+.2f} puntos")
        payload.update({
            "comparado_contra": str(args.compare),
            "comparado_sha256_16": file_hash(args.compare),
            "acuerdo_pixel": round(agree, 5),
            "delta_miou": round(m_b.miou - m.miou, 6),
            "metrics_comparado": m_b.as_dict(nombres),
        })

    if not args.no_json:
        out = _resolve_out_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Guardamos también la matriz cruda para poder recalcular sin re-evaluar.
        np.save(out.with_suffix(".confusion.npy"), conf)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"\n  → {out}")
        print(f"  → {out.with_suffix('.confusion.npy')}")
        print("  generate_report.py levanta el JSON más reciente de outputs/metrics/.")
    return 0


def _resolve_out_path(raw: str | None) -> Path:
    """
    Normaliza ``--out``.

    Si apunta a una carpeta (o a una ruta sin extensión), se agrega el nombre
    versionado ``val_<timestamp>.json`` dentro. Antes, ``--out outputs/metrics``
    escribía un ARCHIVO llamado ``outputs/metrics`` (sin .json) que la siguiente
    corrida pisaba: se perdía el histórico y el `.gitignore`/`generate_report`
    no lo encontraban.
    """
    stamp = f"val_{datetime.now():%Y%m%d_%H%M%S}.json"
    if not raw:
        return PROJ.METRICS_DIR / stamp
    p = Path(raw)
    # Con extensión = archivo explícito; sin extensión = carpeta (versionado).
    return p if p.suffix else p / stamp


if __name__ == "__main__":
    raise SystemExit(main())
