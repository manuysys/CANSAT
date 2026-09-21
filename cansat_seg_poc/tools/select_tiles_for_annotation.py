"""
Cola de anotación (active learning) sobre frames de una misión — `cansat/active.py`.

Escanea una carpeta de frames, corre el modelo de terreno y rankea qué tiles
conviene anotar primero cuando existan labels de vuelo (entropía + rareza +
estrés). No copia imágenes: escribe un CSV ordenado con la ruta y el motivo.

Uso:
    python tools/select_tiles_for_annotation.py --folder entrega/frames --max 50
    python tools/select_tiles_for_annotation.py --folder dataset/loveda_raw/Val/Urban/images_png \
        --onnx outputs/cansat_seg_terrain_v2_224.onnx --max 20

Salida por defecto: dataset/active_learning_queue/<carpeta>/candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402
from cansat import active as ACT                              # noqa: E402
from cansat import ood as OOD                                 # noqa: E402
from cansat import onnxio                                     # noqa: E402
from cansat import preprocess as PP                           # noqa: E402
from cansat import stress as ST                               # noqa: E402

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
STRESS_MAX_SIDE = 320.0


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=0, keepdims=True))
    return e / np.maximum(e.sum(axis=0, keepdims=True), 1e-12)


def analizar_frame(sess, path: Path, img_size: int,
                   clases_prom: list[float] | None) -> dict | None:
    """Métricas de un frame + score de cola. ``None`` si no se pudo leer."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    tensor = PP.preprocess_bgr(bgr, img_size)
    # onnxio.run() ya devuelve (C, H, W): un solo [0] para el eje batch.
    logits = sess.run({"input": tensor})[0]
    probs = softmax(logits)
    ent = ACT.entropia_normalizada(probs)
    pcts = probs.reshape(probs.shape[0], -1).mean(axis=1)
    rar = ACT.rareza(pcts, clases_prom) if clases_prom else 0.0
    h, w = bgr.shape[:2]
    esc = STRESS_MAX_SIDE / max(h, w)
    small = (cv2.resize(bgr, (max(1, int(w * esc)), max(1, int(h * esc))),
                        interpolation=cv2.INTER_AREA) if esc < 1.0 else bgr)
    stress = min(1.0, ST.haze_metrics(small)["haze_pct"] / max(ST.CONTAM_DENSA, 1.0))
    score = ACT.puntaje(ent, rar, stress)
    return {
        "ruta": str(path),
        "entropia": round(ent, 4),
        "rareza": round(rar, 4),
        "stress": round(stress, 4),
        "puntaje": score,
        "motivos": "; ".join(ACT.motivos(ent, rar, stress)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cola de anotación (active learning)")
    ap.add_argument("--folder", required=True, help="carpeta de frames a priorizar")
    ap.add_argument("--onnx", default="outputs/cansat_seg_terrain_v2_224.onnx")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max", type=int, default=50, help="tamaño de la cola")
    ap.add_argument("--out", default=None,
                    help="CSV de salida (default: dataset/active_learning_queue/<carpeta>/candidates.csv)")
    args = ap.parse_args(argv)

    carpeta = Path(args.folder)
    if not carpeta.is_dir():
        print(f"[ERROR] no existe la carpeta: {carpeta}")
        return 1
    frames = [p for p in sorted(carpeta.iterdir())
              if p.suffix.lower() in EXTS and p.is_file()]
    if not frames:
        print(f"[ERROR] no hay imágenes en {carpeta}")
        return 1

    ref = OOD.cargar_referencia(ROOT / OOD.REFERENCIA_DEFECTO)
    clases_prom = ref.get("clases_prom") if ref else None
    if clases_prom is None:
        print("  [WARN] sin referencia OOD: la rareza queda en 0 (entropía y estrés siguen)")

    sess = onnxio.load_required(args.onnx, "terreno (cola de anotación)")
    size = sess.size_px or args.img_size
    if size != args.img_size:
        print(f"  [i] el ONNX fija la entrada a {size}px; se ignora --img-size")
    filas = [f for p in frames if (f := analizar_frame(sess, p, size, clases_prom))]
    filas.sort(key=lambda f: -f["puntaje"])
    cola = filas[: args.max]

    out = Path(args.out) if args.out else (
        ROOT / "dataset" / "active_learning_queue" / carpeta.name / "candidates.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ruta", "entropia", "rareza", "stress",
                                           "puntaje", "motivos"])
        w.writeheader()
        w.writerows(cola)

    print(f"  {len(filas)} frames analizados · cola de {len(cola)} en {out}")
    for f in cola[:5]:
        print(f"    {f['puntaje']:.3f}  {Path(f['ruta']).name}  ({f['motivos']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
